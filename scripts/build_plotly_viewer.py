"""Static HTML scatter explorer powered by Plotly + raw JS.

Per dataset, writes:
    viewer/{dataset}/
        index.html              ← Plotly scatter + filters + bottom detail panel
        frames/{i}.png          ← frame_grid thumbnail per segment (5 frames, 64px)
        frames_big/{i}.png      ← larger frame_grid for the detail panel (5 × 128px)
        videos/{i}.mp4          ← H.264 segment video (cv2 + ffmpeg transcode)

UX (rendered in browser):
    • Scatter colored by cluster, legend filters per-cluster.
    • Hover → tooltip with task / cluster / seg_len  + a thumbnail preview in
      a side card that updates on hover (no clipping, no fixed tooltip size).
    • Click point → bottom detail panel populates with big frame_grid + a
      <video controls autoplay loop> for the segment + all metadata.
    • Top filters: task dropdown, cluster dropdown.

Run inside the container (needs cv2, h5py, system ffmpeg):

    docker exec -it lotus_test bash -c '
      cd /home/iw/lotus_test && \
      python scripts/build_plotly_viewer.py \
          --datasets libero_object libero_goal robocasa365_atomic_seen \
          --out-dir viewer \
          --include-video'

Then host the viewer/ directory locally:

    cd viewer && python -m http.server 8000
    # then open http://localhost:8000/  in a browser
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering


def relabel_segments(segs: dict, method: str) -> dict:
    """Optionally replace the saved (LOTUS) cluster labels with another
    skill-discovery method's clustering, computed on the SAME DINOv2
    embeddings at the SAME number of clusters K. Lets the explorer be
    colored by e.g. BUDS instead of LOTUS. 'lotus' = keep saved labels."""
    if method == "lotus":
        return segs
    X = segs["embeddings"].astype(np.float64)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    K = int(segs["cluster_labels"].max()) + 1
    if method == "buds":
        labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Xn)
    else:
        raise ValueError(f"unknown clustering method: {method}")
    segs = dict(segs)
    segs["cluster_labels"] = labels.astype(segs["cluster_labels"].dtype)
    return segs


# Same registry as wandb_skill_explorer.py
DATASETS = {
    "libero_object": {
        "exp_dir": "lotus/skill_learning/results/dinov2_libero_object_image_only_10",
        "raw_root": "datasets/libero_object",
        "category": "libero_object",
        "flip_v": True,
    },
    "libero_goal": {
        "exp_dir": "lotus/skill_learning/results/dinov2_libero_goal_image_only_10",
        "raw_root": "datasets/libero_goal",
        "category": "libero_goal",
        "flip_v": True,
    },
    "robocasa365_atomic_seen": {
        "exp_dir": "lotus/skill_learning/results/dinov2_robocasa365_atomic_seen_image_only_18",
        "raw_root": "datasets/robocasa365_atomic_seen",
        "category": "robocasa365_atomic_seen",
        "flip_v": False,
    },
}


def short_task_name(name: str) -> str:
    n = name
    n = n.replace("pick_up_the_", "").replace("_and_place_it_in_the_basket_demo", "")
    if n.endswith("_demo"):
        n = n[: -len("_demo")]
    return n


def load_segments(exp_dir: str) -> dict:
    with h5py.File(Path(exp_dir) / "skill_data" / "saved_feature_data.hdf5", "r") as f:
        return {
            "embeddings": f["embeddings"][()],
            "cluster_labels": f["cluster_labels"][()],
            "task_ids": f["task_ids"][()],
            "demo_indices": f["demo_indices"][()],
            "seg_start": f["seg_start"][()],
            "seg_end": f["seg_end"][()],
        }


def list_task_files(exp_dir: str, category: str) -> list[str]:
    skill_dir = Path(exp_dir) / "skill_data" / category
    return [f.name.split("_subtasks_", 1)[0]
            for f in sorted(skill_dir.glob("*_subtasks_*.hdf5"))]


def read_segment_frames(raw_path: Path, demo_idx: int, start: int, end: int,
                        n: int, image_size: int, flip_v: bool = False):
    with h5py.File(raw_path, "r") as f:
        imgs = f[f"data/demo_{demo_idx}/obs/agentview_rgb"]
        T = imgs.shape[0]
        s = max(0, min(start, T - 1))
        e = max(s, min(end, T - 1))
        idxs = np.linspace(s, e, n).astype(int)
        frames = np.stack([imgs[i] for i in idxs], axis=0)
    if frames.shape[-1] != 3 and frames.shape[1] == 3:
        frames = frames.transpose(0, 2, 3, 1)
    if flip_v:
        frames = frames[:, ::-1, :, :]
    if image_size and frames.shape[1] != image_size:
        frames = np.stack(
            [cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_AREA)
             for f in frames], axis=0
        )
    return frames


def make_frame_grid(frames: np.ndarray) -> np.ndarray:
    return np.concatenate(list(frames), axis=1)


def write_video(raw_path: Path, demo_idx: int, start: int, end: int,
                image_size: int, every: int, fps: int, out_path: Path,
                flip_v: bool = False):
    with h5py.File(raw_path, "r") as f:
        imgs = f[f"data/demo_{demo_idx}/obs/agentview_rgb"]
        T = imgs.shape[0]
        s = max(0, min(start, T - 1))
        e = max(s, min(end, T - 1))
        idxs = list(range(s, e + 1, every))
        if not idxs or idxs[-1] != e:
            idxs.append(e)
        frames = np.stack([imgs[i] for i in idxs], axis=0)
    if flip_v:
        frames = frames[:, ::-1, :, :]
    if image_size and frames.shape[1] != image_size:
        frames = np.stack([cv2.resize(f, (image_size, image_size)) for f in frames], axis=0)
    H, W = frames.shape[1], frames.shape[2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        return None
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out_path)],
            check=True, timeout=30,
        )
    except Exception as ex:
        print(f"  [warn] ffmpeg transcode failed: {ex}")
        tmp.replace(out_path)
        return out_path
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return out_path


def get_task_instruction(raw_path: Path, demo_idx: int) -> str:
    try:
        with h5py.File(raw_path, "r") as f:
            attrs = f[f"data/demo_{demo_idx}"].attrs
            if "task_instruction" in attrs:
                return str(attrs["task_instruction"])
    except Exception:
        pass
    return ""


# ── HTML template ──
HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 14px; background: #fafafa; color: #222; }
  h1 { margin: 0 0 8px; font-size: 18px; }
  .meta-top { color: #666; font-size: 13px; margin-bottom: 12px; }
  .layout { display: grid; grid-template-columns: 1fr 280px; gap: 14px; }
  #plot { background: #fff; border: 1px solid #ddd; border-radius: 6px;
          height: 700px; min-width: 0; }
  .hover-card { background: #fff; border: 1px solid #ddd; border-radius: 6px;
                padding: 10px; height: fit-content; position: sticky; top: 12px; }
  .hover-card h3 { margin: 0 0 8px; font-size: 13px; color: #666;
                   text-transform: uppercase; letter-spacing: 0.5px; }
  .hover-card img { width: 100%; max-width: 260px; image-rendering: auto;
                    border-radius: 4px; background: #f0f0f0; }
  .hover-card .meta { font-size: 12px; margin-top: 8px; color: #333; line-height: 1.5; }
  .hover-card .meta b { color: #111; }
  .filters { display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
             margin-bottom: 10px; }
  .filters label { font-size: 13px; color: #444; }
  .filters select { padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px;
                    background: #fff; font-size: 13px; min-width: 180px; }
  .filters button { padding: 4px 10px; border: 1px solid #ccc; border-radius: 4px;
                    background: #fff; cursor: pointer; font-size: 13px; }
  .filters button:hover { background: #f0f0f0; }
  #detail { display: grid; grid-template-columns: 380px 1fr; gap: 18px;
            background: #fff; border: 1px solid #ddd; border-radius: 6px;
            padding: 16px; margin-top: 14px; min-height: 220px; }
  #detail.empty { color: #999; font-style: italic; }
  #detail .media h4 { margin: 0 0 6px; font-size: 12px; color: #666;
                       text-transform: uppercase; letter-spacing: 0.5px; }
  #detail .media img,
  #detail .media video { width: 100%; max-width: 360px; border-radius: 4px;
                         background: #f0f0f0; }
  #detail .meta dl { display: grid; grid-template-columns: 130px 1fr;
                     gap: 4px 12px; font-size: 13px; margin: 0; }
  #detail .meta dt { color: #888; }
  #detail .meta dd { margin: 0; color: #111; word-break: break-word; }
  #detail .meta dt.bigval { font-weight: 600; }
  #detail .meta .pill { display: inline-block; padding: 1px 8px; border-radius: 999px;
                        background: #eee; font-size: 12px; }
</style>
</head>
<body>

<h1>__TITLE__</h1>
<div class="meta-top">__SUBTITLE__</div>

<div class="filters">
  <label>Task: <select id="task-filter"><option value="">(all)</option></select></label>
  <label>Cluster: <select id="cluster-filter"><option value="">(all)</option></select></label>
  <button onclick="resetFilters()">Reset</button>
  <span id="visible-count" style="color:#666; font-size:12px;"></span>
</div>

<div class="layout">
  <div id="plot"></div>
  <div class="hover-card" id="hover-card">
    <h3>Hover preview</h3>
    <img id="hover-img" src="" alt="hover preview" />
    <div class="meta" id="hover-meta">
      Hover over a point to see its frame grid here.
    </div>
  </div>
</div>

<div id="detail" class="empty">
  Click a point to load its full preview (large frame grid + video).
</div>

<script>
const DATA = __DATA_JSON__;
const N = DATA.length;
const TASKS = __TASKS_JSON__;        // [task_short, ...] indexed by task_idx
const TASK_FULL = __TASK_FULL_JSON__; // full task names
const CATEGORY = "__CATEGORY__";
const K = __K__;
const INCLUDE_VIDEO = __INCLUDE_VIDEO__;

// Categorical palette (Plotly D3) — repeats if K>10
const PAL = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2",
             "#7f7f7f","#bcbd22","#17becf","#e6194B","#3cb44b","#4363d8","#f58231",
             "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#dcbeff",
             "#9A6324","#fffac8","#800000","#aaffc3","#808000","#ffd8b1","#000075",
             "#a9a9a9","#000000","#ff0000","#00ff00","#0000ff","#ffff00","#00ffff",
             "#ff00ff","#c0c0c0","#404040","#ffb700","#8b008b"];

// Build per-cluster traces (legend filters per-cluster)
function makeTraces(visible_mask) {
  const traces = [];
  for (let c = 0; c < K; c++) {
    const xs = [], ys = [], custom = [], txts = [];
    for (let i = 0; i < N; i++) {
      if (DATA[i].cluster_idx !== c) continue;
      if (visible_mask && !visible_mask[i]) continue;
      xs.push(DATA[i].tsne_x);
      ys.push(DATA[i].tsne_y);
      custom.push(i);
      txts.push(
        `<b>skill ${c}</b><br>` +
        `task: ${TASKS[DATA[i].task_idx]}<br>` +
        `demo: ${DATA[i].demo_idx}<br>` +
        `range: [${DATA[i].seg_start}, ${DATA[i].seg_end}] (len ${DATA[i].seg_len})`
      );
    }
    traces.push({
      x: xs, y: ys,
      mode: "markers",
      type: "scatter",
      name: `skill ${c}`,
      marker: { color: PAL[c % PAL.length], size: 6, opacity: 0.75,
                line: { color: "white", width: 0.4 } },
      customdata: custom,
      text: txts,
      hovertemplate: "%{text}<extra></extra>",
      hoverlabel: { bgcolor: "white", bordercolor: PAL[c % PAL.length] },
    });
  }
  return traces;
}

const layout = {
  margin: { l: 50, r: 20, t: 20, b: 50 },
  xaxis: { title: "t-SNE 1", zeroline: false, gridcolor: "#eee" },
  yaxis: { title: "t-SNE 2", zeroline: false, gridcolor: "#eee" },
  hovermode: "closest",
  legend: { orientation: "v", x: 1.02, y: 1, xanchor: "left",
            font: { size: 11 } },
  plot_bgcolor: "#fff", paper_bgcolor: "#fff",
};

const config = { responsive: true, displaylogo: false,
                 modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d"] };

Plotly.newPlot("plot", makeTraces(null), layout, config);

const plotEl = document.getElementById("plot");
const hoverImg = document.getElementById("hover-img");
const hoverMeta = document.getElementById("hover-meta");
const detail = document.getElementById("detail");

function fmtRow(i) {
  const d = DATA[i];
  return {
    idx: i,
    task: TASKS[d.task_idx],
    task_full: TASK_FULL[d.task_idx],
    cluster: d.cluster_idx,
    demo: d.demo_idx,
    s: d.seg_start, e: d.seg_end, len: d.seg_len,
    instr: d.instr || "",
  };
}

plotEl.on("plotly_hover", (ev) => {
  const i = ev.points[0].customdata;
  const r = fmtRow(i);
  hoverImg.src = `frames/${i}.png`;
  hoverMeta.innerHTML =
    `<b>skill ${r.cluster}</b> · ${r.task}<br>` +
    `demo ${r.demo}, len ${r.len}<br>` +
    (r.instr ? `<i>${r.instr.replace(/</g,'&lt;')}</i>` : "");
});

plotEl.on("plotly_click", (ev) => {
  const i = ev.points[0].customdata;
  const r = fmtRow(i);
  detail.classList.remove("empty");
  const vidHtml = INCLUDE_VIDEO
    ? `<div class="media"><h4>Segment video</h4>
         <video src="videos/${i}.mp4" controls autoplay loop muted playsinline></video></div>`
    : "";
  detail.innerHTML =
    `<div class="media"><h4>Frame grid (first → last)</h4>
       <img src="frames_big/${i}.png" alt="frame grid" />
       ${vidHtml}
     </div>
     <div class="meta">
       <dl>
         <dt>skill cluster</dt><dd><span class="pill" style="background:${PAL[r.cluster % PAL.length]};color:white;">skill ${r.cluster}</span></dd>
         <dt>task</dt><dd>${r.task_full}</dd>
         <dt>demo idx</dt><dd>${r.demo}</dd>
         <dt>frame range</dt><dd>[${r.s} … ${r.e}]  (length ${r.len})</dd>
         <dt>t-SNE coord</dt><dd>(${DATA[i].tsne_x.toFixed(2)}, ${DATA[i].tsne_y.toFixed(2)})</dd>
         ${r.instr ? `<dt>instruction</dt><dd><i>${r.instr.replace(/</g,'&lt;')}</i></dd>` : ""}
         <dt>segment idx</dt><dd>${i}</dd>
       </dl>
     </div>`;
});

// ── Filters ──
const taskFilter = document.getElementById("task-filter");
const clusterFilter = document.getElementById("cluster-filter");
const visCount = document.getElementById("visible-count");

TASKS.forEach((t, idx) => {
  const o = document.createElement("option");
  o.value = idx; o.textContent = t;
  taskFilter.appendChild(o);
});
for (let c = 0; c < K; c++) {
  const o = document.createElement("option");
  o.value = c; o.textContent = `skill ${c}`;
  clusterFilter.appendChild(o);
}

function applyFilters() {
  const tf = taskFilter.value === "" ? null : parseInt(taskFilter.value);
  const cf = clusterFilter.value === "" ? null : parseInt(clusterFilter.value);
  const mask = new Array(N);
  let count = 0;
  for (let i = 0; i < N; i++) {
    const d = DATA[i];
    const ok = (tf === null || d.task_idx === tf) &&
               (cf === null || d.cluster_idx === cf);
    mask[i] = ok;
    if (ok) count++;
  }
  visCount.textContent = `${count} / ${N} segments visible`;
  Plotly.react("plot", makeTraces(mask), layout, config);
}

function resetFilters() {
  taskFilter.value = "";
  clusterFilter.value = "";
  applyFilters();
}

taskFilter.addEventListener("change", applyFilters);
clusterFilter.addEventListener("change", applyFilters);
applyFilters();
</script>
</body>
</html>
"""


def build_dataset(name: str, cfg: dict, out_root: Path, image_size: int,
                  big_image_size: int, frames_per_grid: int,
                  include_video: bool, video_image_size: int, video_every: int,
                  fps: int, clustering: str = "lotus"):
    print(f"\n=== {name}  (clustering={clustering}) ===")
    segs = load_segments(cfg["exp_dir"])
    segs = relabel_segments(segs, clustering)
    task_names = list_task_files(cfg["exp_dir"], cfg["category"])
    task_shorts = [short_task_name(t) for t in task_names]
    n = len(segs["embeddings"])
    K = int(segs["cluster_labels"].max()) + 1
    print(f"  segments={n}  tasks={len(task_names)}  K={K}")

    perplexity = min(30, max(5, n // 4))
    print(f"  t-SNE (perplexity={perplexity})...", flush=True)
    t0 = time.time()
    pts = TSNE(n_components=2, random_state=42, perplexity=perplexity,
               init="pca").fit_transform(segs["embeddings"])
    print(f"    done {time.time()-t0:.1f}s")

    ds_dir = out_root / name
    (ds_dir / "frames").mkdir(parents=True, exist_ok=True)
    (ds_dir / "frames_big").mkdir(parents=True, exist_ok=True)
    if include_video:
        (ds_dir / "videos").mkdir(parents=True, exist_ok=True)

    raw_root = Path(cfg["raw_root"])
    flip_v = bool(cfg.get("flip_v", False))
    rows = []
    failed = 0
    t0 = time.time()
    for i in range(n):
        tid = int(segs["task_ids"][i])
        cid = int(segs["cluster_labels"][i])
        ep = int(segs["demo_indices"][i])
        s = int(segs["seg_start"][i])
        e = int(segs["seg_end"][i])
        task = task_names[tid] if 0 <= tid < len(task_names) else f"task{tid}"
        raw_path = raw_root / f"{task}.hdf5"
        instr = get_task_instruction(raw_path, ep)

        # Small thumbnail for hover preview + scatter side card
        try:
            fr = read_segment_frames(raw_path, ep, s, e, frames_per_grid,
                                      image_size, flip_v=flip_v)
            grid = make_frame_grid(fr)
            cv2.imwrite(str(ds_dir / "frames" / f"{i}.png"),
                        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        except Exception as ex:
            failed += 1
            print(f"  [warn] small grid failed seg{i} {task} ep{ep}: {ex}")

        # Larger frame grid for the bottom detail panel
        try:
            fr = read_segment_frames(raw_path, ep, s, e, frames_per_grid,
                                      big_image_size, flip_v=flip_v)
            grid = make_frame_grid(fr)
            cv2.imwrite(str(ds_dir / "frames_big" / f"{i}.png"),
                        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        except Exception as ex:
            failed += 1

        if include_video:
            try:
                write_video(raw_path, ep, s, e, video_image_size, video_every,
                            fps=fps, out_path=ds_dir / "videos" / f"{i}.mp4",
                            flip_v=flip_v)
            except Exception as ex:
                failed += 1
                print(f"  [warn] video failed seg{i}: {ex}")

        rows.append({
            "task_idx": tid,
            "cluster_idx": cid,
            "demo_idx": ep,
            "seg_start": s,
            "seg_end": e,
            "seg_len": e - s,
            "tsne_x": float(pts[i, 0]),
            "tsne_y": float(pts[i, 1]),
            "instr": instr,
        })

        if (i + 1) % 100 == 0 or i == n - 1:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            print(f"    {i+1}/{n}  {rate:.1f} seg/s  eta={eta:.0f}s  failed={failed}")

    # Render HTML
    method_label = {"lotus": "LOTUS spectral",
                    "buds": "BUDS agglomerative (Ward)"}.get(clustering, clustering)
    method_short = {"lotus": "LOTUS", "buds": "BUDS"}.get(clustering, clustering.upper())
    title = f"{name} — {method_short} skill explorer"
    subtitle = (f"{n} segments · {len(task_names)} tasks · K={K} "
                f"({method_label} clustering on DINOv2-1536 segment features"
                + (", LOTUS segmentation" if clustering != "lotus" else "")
                + ", t-SNE projection).")
    html = (HTML_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__DATA_JSON__", json.dumps(rows))
            .replace("__TASKS_JSON__", json.dumps(task_shorts))
            .replace("__TASK_FULL_JSON__", json.dumps(task_names))
            .replace("__CATEGORY__", cfg["category"])
            .replace("__K__", str(K))
            .replace("__INCLUDE_VIDEO__", "true" if include_video else "false"))
    (ds_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"  ✅ wrote {ds_dir/'index.html'}")

    return {"name": name, "n": n, "K": K, "n_tasks": len(task_names),
            "subtitle": subtitle}


def write_index_page(out_root: Path, summaries: list[dict], clustering: str = "lotus"):
    method_short = {"lotus": "LOTUS", "buds": "BUDS"}.get(clustering, clustering.upper())
    rows = []
    for s in summaries:
        rows.append(
            f'<li><a href="{s["name"]}/index.html">{s["name"]}</a>'
            f' — {s["n"]} segments, {s["n_tasks"]} tasks, K={s["K"]}</li>'
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8' />"
        f"<title>{method_short} skill explorer</title>"
        "<style>body{font-family:-apple-system,Segoe UI,sans-serif;"
        "max-width:720px;margin:24px auto;padding:0 16px;color:#222;}"
        "h1{margin:0 0 16px}li{margin:6px 0;font-size:15px}"
        "a{color:#1f6fb0;text-decoration:none}a:hover{text-decoration:underline}"
        "p{color:#666}</style></head><body>"
        f"<h1>{method_short} skill discovery — scatter explorers</h1>"
        "<p>Click a dataset to open its interactive t-SNE scatter. "
        "Hover a point for a thumbnail preview; click for the full "
        "frame grid + segment video.</p><ul>"
        + "\n".join(rows) +
        "</ul></body></html>"
    )
    (out_root / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {out_root/'index.html'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--out-dir", type=Path, default=Path("viewer"))
    ap.add_argument("--image-size", type=int, default=64,
                    help="Side-card thumbnail size (px per frame).")
    ap.add_argument("--big-image-size", type=int, default=128,
                    help="Detail-panel frame grid size (px per frame).")
    ap.add_argument("--frames-per-grid", type=int, default=5)
    ap.add_argument("--include-video", action="store_true")
    ap.add_argument("--video-size", type=int, default=160)
    ap.add_argument("--video-every", type=int, default=4)
    ap.add_argument("--video-fps", type=int, default=10)
    ap.add_argument("--clustering", choices=["lotus", "buds"], default="lotus",
                    help="Color points by saved LOTUS labels or recomputed BUDS (Ward) at same K.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in args.datasets:
        cfg = DATASETS.get(name)
        if cfg is None:
            print(f"unknown dataset {name!r}")
            continue
        if not Path(cfg["exp_dir"]).exists():
            print(f"[skip] {name}: {cfg['exp_dir']} missing")
            continue
        s = build_dataset(
            name, cfg, args.out_dir,
            image_size=args.image_size,
            big_image_size=args.big_image_size,
            frames_per_grid=args.frames_per_grid,
            include_video=args.include_video,
            video_image_size=args.video_size,
            video_every=args.video_every,
            fps=args.video_fps,
            clustering=args.clustering,
        )
        summaries.append(s)

    write_index_page(args.out_dir, summaries, clustering=args.clustering)
    print("done.")


if __name__ == "__main__":
    main()
