"""WandB skill explorer: upload an interactive table of every (segment) point
with its task/cluster labels, t-SNE coordinates, and a thumbnail/video of the
underlying frames.

After uploading you can:
  1. Add a Scatter Chart panel from tsne_x / tsne_y, colored by `cluster_idx`
     (or `task_idx`), faceted by `dataset` — the same plot we generated as
     a static PNG, but interactive.
  2. Click a point → side panel shows the row → see frame_grid + segment_video,
     plus task name / demo id / segment range.
  3. Filter by `dataset`, `task_short`, or `cluster_idx`.

Run inside the container (where the raw datasets and skill_data live):

    docker exec -it lotus_test wandb login         # one-time
    docker exec -it lotus_test bash -c '
      cd /home/iw/lotus_test && \
      python scripts/wandb_skill_explorer.py \
          --datasets libero_object libero_goal robocasa365_atomic_seen \
          --include-video \
          --project lotus-skill-explorer'

Storage tips:
  --image-size  controls the frame_grid thumbnail size (default 64, 5 frames
                horizontally → ~320×64 PNG ≈ 8 KB/segment).
  --include-video to additionally embed an mp4 per segment (~1.5 MB each,
                so 2700 segments ≈ 4 GB upload — slow but very useful).
  --max-segments to subsample (each dataset capped to N) for a quick demo.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import wandb
from sklearn.manifold import TSNE


# ─────────── dataset registry ───────────
# `flip_v` is True for sources that store frames in OpenGL convention (origin
# bottom-left, vertically flipped vs the natural top-down view) — LIBERO's
# robosuite simulator does this. RoboCasa365 was converted from regular mp4
# so it's already top-down.
DATASETS = {
    "libero_object": {
        "exp_dir": "lotus/skill_learning/results/dinov2_libero_object_image_only_10",
        "raw_root": "datasets/libero_object",
        "category": "libero_object",
        "n_tasks": 10,
        "flip_v": True,
    },
    "libero_goal": {
        "exp_dir": "lotus/skill_learning/results/dinov2_libero_goal_image_only_10",
        "raw_root": "datasets/libero_goal",
        "category": "libero_goal",
        "n_tasks": 10,
        "flip_v": True,
    },
    "robocasa365_atomic_seen": {
        "exp_dir": "lotus/skill_learning/results/dinov2_robocasa365_atomic_seen_image_only_18",
        "raw_root": "datasets/robocasa365_atomic_seen",
        "category": "robocasa365_atomic_seen",
        "n_tasks": 18,
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
    feat = Path(exp_dir) / "skill_data" / "saved_feature_data.hdf5"
    with h5py.File(feat, "r") as f:
        return {
            "embeddings": f["embeddings"][()],
            "cluster_labels": f["cluster_labels"][()],
            "task_ids": f["task_ids"][()],
            "demo_indices": f["demo_indices"][()],
            "seg_start": f["seg_start"][()],
            "seg_end": f["seg_end"][()],
        }


def list_task_files(exp_dir: str, category: str) -> list[str]:
    """The visualize_clustering.py mapping: task_id = position of the task's
    subtask hdf5 in the sorted listing under skill_data/{category}/."""
    skill_dir = Path(exp_dir) / "skill_data" / category
    files = sorted(skill_dir.glob("*_subtasks_*.hdf5"))
    return [f.name.split("_subtasks_", 1)[0] for f in files]


def read_segment_frames(raw_path: Path, demo_idx: int, start: int, end: int,
                        n: int, image_size: int, flip_v: bool = False):
    """Return (n, H, W, 3) uint8 frames sampled evenly from [start, end]."""
    with h5py.File(raw_path, "r") as f:
        imgs = f[f"data/demo_{demo_idx}/obs/agentview_rgb"]
        T = imgs.shape[0]
        s = max(0, min(start, T - 1))
        e = max(s, min(end, T - 1))
        idxs = np.linspace(s, e, n).astype(int)
        frames = np.stack([imgs[i] for i in idxs], axis=0)
    if frames.shape[-1] != 3 and frames.shape[1] == 3:  # CHW
        frames = frames.transpose(0, 2, 3, 1)
    if flip_v:
        frames = frames[:, ::-1, :, :]  # vertical flip
    if image_size and frames.shape[1] != image_size:
        frames = np.stack([
            cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_AREA)
            for f in frames
        ], axis=0)
    return frames


def make_frame_grid(frames: np.ndarray) -> np.ndarray:
    """Stack frames horizontally → (H, W*n, 3) uint8."""
    return np.concatenate(list(frames), axis=1)


def make_segment_video_file(raw_path: Path, demo_idx: int, start: int, end: int,
                            image_size: int, every: int, fps: int, out_path: Path,
                            flip_v: bool = False):
    """Encode the segment as an mp4 file. We write mp4v with cv2 (which works
    everywhere), then transcode to H.264 with ffmpeg so the WandB browser
    player can decode it (cv2's bundled ffmpeg lacks the libx264 encoder,
    but the system ffmpeg has it).

    Returns the path written, or None on failure.
    """
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
        frames = np.stack(
            [cv2.resize(f, (image_size, image_size)) for f in frames], axis=0
        )
    H, W = frames.shape[1], frames.shape[2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 1) Write intermediate mp4v
    tmp_path = out_path.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    if not writer.isOpened():
        return None
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()

    # 2) Transcode to H.264 (browser-playable)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(tmp_path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             str(out_path)],
            check=True, timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as ex:
        print(f"  [warn] ffmpeg transcode failed: {ex}")
        # Fall back to the mp4v file
        tmp_path.replace(out_path)
        return out_path
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return out_path


def get_task_instruction(raw_path: Path, demo_idx: int) -> str:
    """RoboCasa365 hdf5 stores per-demo language instruction; LIBERO doesn't."""
    try:
        with h5py.File(raw_path, "r") as f:
            attrs = f[f"data/demo_{demo_idx}"].attrs
            if "task_instruction" in attrs:
                return str(attrs["task_instruction"])
    except Exception:
        pass
    return ""


def process_dataset(name: str, cfg: dict, max_segments: Optional[int],
                    image_size: int, video_image_size: int, video_every: int,
                    include_video: bool, frames_per_grid: int):
    print(f"\n=== {name} ===")
    segs = load_segments(cfg["exp_dir"])
    task_names = list_task_files(cfg["exp_dir"], cfg["category"])
    n_total = len(segs["embeddings"])
    K = int(segs["cluster_labels"].max()) + 1
    print(f"  segments={n_total}  tasks={len(task_names)}  K={K}")

    # Optional subsample (deterministic)
    idx = np.arange(n_total)
    if max_segments is not None and n_total > max_segments:
        idx = np.random.RandomState(0).choice(n_total, max_segments, replace=False)
        idx.sort()
        for k in segs:
            segs[k] = segs[k][idx]
        print(f"  subsampled to {len(idx)}")
    n = len(idx)

    # t-SNE
    perplexity = min(30, max(5, n // 4))
    print(f"  t-SNE (perplexity={perplexity})...", flush=True)
    t0 = time.time()
    pts = TSNE(n_components=2, random_state=42, perplexity=perplexity,
               init="pca").fit_transform(segs["embeddings"])
    print(f"    done {time.time()-t0:.1f}s")

    # Build table
    cols = [
        "dataset", "task_idx", "task_name", "task_short",
        "cluster_idx", "tsne_x", "tsne_y",
        "demo_idx", "seg_start", "seg_end", "seg_len",
        "task_instruction", "frame_grid",
    ]
    if include_video:
        cols.append("segment_video")
    table = wandb.Table(columns=cols)

    raw_root = Path(cfg["raw_root"])
    flip_v = bool(cfg.get("flip_v", False))
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

        try:
            frames = read_segment_frames(raw_path, ep, s, e, frames_per_grid,
                                          image_size, flip_v=flip_v)
            grid = make_frame_grid(frames)
            grid_obj = wandb.Image(grid, caption=f"{task} ep{ep} [{s}-{e}]")
        except Exception as ex:
            failed += 1
            print(f"  [warn] grid failed: {task} ep{ep} {s}-{e}: {ex}")
            grid_obj = wandb.Image(np.zeros((image_size, image_size * frames_per_grid, 3), np.uint8))

        row = [
            name, tid, task, short_task_name(task),
            cid, float(pts[i, 0]), float(pts[i, 1]),
            ep, s, e, e - s,
            instr, grid_obj,
        ]
        if include_video:
            vid_obj = None
            try:
                vid_path = Path(f"/tmp/wandb_videos/{name}_{tid}_{ep}_{s}_{e}.mp4")
                ok = make_segment_video_file(
                    raw_path, ep, s, e, video_image_size, video_every,
                    fps=10, out_path=vid_path, flip_v=flip_v,
                )
                if ok is not None and vid_path.exists():
                    vid_obj = wandb.Video(str(vid_path), fps=10, format="mp4")
            except Exception as ex:
                failed += 1
                print(f"  [warn] video failed: {task} ep{ep} {s}-{e}: {ex}")
            row.append(vid_obj)

        table.add_data(*row)

        if (i + 1) % 100 == 0 or i == n - 1:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            print(f"    {i+1}/{n}  {rate:.1f} seg/s  eta={eta:.0f}s  failed={failed}")

    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="lotus-skill-explorer")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    help="Subset of datasets to upload")
    ap.add_argument("--max-segments", type=int, default=None,
                    help="Per-dataset cap (for quick demo). Default: all.")
    ap.add_argument("--image-size", type=int, default=64,
                    help="Frame grid thumbnail size (default 64).")
    ap.add_argument("--frames-per-grid", type=int, default=5)
    ap.add_argument("--include-video", action="store_true",
                    help="Embed mp4 per segment (~1.5 MB each).")
    ap.add_argument("--video-size", type=int, default=128)
    ap.add_argument("--video-every", type=int, default=4,
                    help="Sample every Nth frame for the video (default 4).")
    args = ap.parse_args()

    # Validate dataset choices
    for d in args.datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset {d!r}; valid: {list(DATASETS)}")

    run_name = args.run_name or "skill_explorer_" + time.strftime("%Y%m%d_%H%M%S")
    print(f"wandb project = {args.project}, run = {run_name}")
    run = wandb.init(project=args.project, name=run_name,
                     config={
                         "image_size": args.image_size,
                         "frames_per_grid": args.frames_per_grid,
                         "include_video": args.include_video,
                         "max_segments": args.max_segments,
                     })

    for name in args.datasets:
        cfg = DATASETS[name]
        if not Path(cfg["exp_dir"]).exists():
            print(f"[skip] {name}: {cfg['exp_dir']} not found")
            continue
        table = process_dataset(
            name, cfg,
            max_segments=args.max_segments,
            image_size=args.image_size,
            video_image_size=args.video_size,
            video_every=args.video_every,
            include_video=args.include_video,
            frames_per_grid=args.frames_per_grid,
        )
        # Log under a unique key so each dataset is its own table in the UI
        wandb.log({f"{name}/segments_table": table})
        print(f"  ✅ logged {name}/segments_table")

    run.finish()
    print("done.")


if __name__ == "__main__":
    main()
