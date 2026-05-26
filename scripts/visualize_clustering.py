"""LOTUS clustering visualization.

Generates three summary figures from results of `agglomoration_script.py`:
  1. Per-task timeline grid: cluster label vs time, 50 demos per task panel.
  2. t-SNE scatter of all segment embeddings, colored by cluster.
  3. t-SNE scatter of all segment embeddings, colored by task.

Designed to run on the libero_object output but takes paths as arguments
so it works on any LOTUS skill_data dir.
"""

import argparse
import glob
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


CLUSTER_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
]


def task_short_name(p):
    name = os.path.basename(p)
    name = re.split(r"_subtasks_", name, maxsplit=1)[0]
    name = re.sub(r"^pick_up_the_", "", name)
    name = re.sub(r"_and_place_it_in_the_basket_demo$", "", name)
    name = re.sub(r"_demo$", "", name)  # libero_goal style
    return name


def collect_subtask_hdf5(exp_dir, category):
    pattern = os.path.join(exp_dir, "skill_data", category, "*.hdf5")
    files = sorted(glob.glob(pattern))
    return files


def parse_K_from_filename(path):
    m = re.search(r"_K(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def render_timeline_grid(subtask_files, out_path, K):
    n = len(subtask_files)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax_idx, fpath in enumerate(subtask_files):
        ax = axes[ax_idx]
        with h5py.File(fpath, "r") as f:
            g = f["subtasks"]
            demo_keys = sorted(
                (k for k in g.keys() if k.startswith("demo_subtasks_seq_")),
                key=lambda k: int(k.rsplit("_", 1)[-1]),
            )
            for k in demo_keys:
                ep_idx = int(k.rsplit("_", 1)[-1])
                segs = g[k][()]
                for start, end, label in segs:
                    ax.plot(
                        [start, end], [ep_idx, ep_idx],
                        color=CLUSTER_COLORS[label % len(CLUSTER_COLORS)],
                        linewidth=2,
                    )

        ax.set_title(task_short_name(fpath), fontsize=10, fontweight="bold")
        ax.set_xlabel("frame")
        ax.set_ylabel("demo")
        ax.set_xlim(left=0)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Line2D([0], [0], color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], lw=4, label=f"skill {i}")
        for i in range(K)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=K, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"LOTUS skill segmentation (K={K})", fontsize=13)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] timeline → {out_path}")


def render_tsne(saved_feature_path, subtask_files, out_path):
    with h5py.File(saved_feature_path, "r") as f:
        emb = f["embeddings"][()]
        cluster_labels = f["cluster_labels"][()]
        task_ids = f["task_ids"][()]
    print(f"segments={len(emb)} dim={emb.shape[1]} K={len(np.unique(cluster_labels))} tasks={len(np.unique(task_ids))}")

    perplexity = min(30, max(5, len(emb) // 4))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
    pts = tsne.fit_transform(emb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for c in np.unique(cluster_labels):
        m = cluster_labels == c
        ax1.scatter(pts[m, 0], pts[m, 1],
                    color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                    label=f"skill {c}", alpha=0.7, s=25, edgecolor="white", linewidth=0.3)
    ax1.set_title("Segment embeddings — colored by skill cluster")
    ax1.legend(loc="best", fontsize=9)
    ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")
    ax1.grid(alpha=0.3)

    task_cmap = plt.get_cmap("tab10")
    task_names = {i: task_short_name(p) for i, p in enumerate(subtask_files)}
    for tid in np.unique(task_ids):
        m = task_ids == tid
        ax2.scatter(pts[m, 0], pts[m, 1],
                    color=task_cmap(tid % 10),
                    label=task_names.get(int(tid), f"task {tid}"),
                    alpha=0.7, s=25, edgecolor="white", linewidth=0.3)
    ax2.set_title("Segment embeddings — colored by source task")
    ax2.legend(loc="best", fontsize=8)
    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")
    ax2.grid(alpha=0.3)

    fig.suptitle("LOTUS skill embeddings (t-SNE of segment features)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] tsne → {out_path}")


def render_aggregate_skill_distribution(subtask_files, out_path, K):
    counts = np.zeros((len(subtask_files), K), dtype=np.int64)
    for ti, fpath in enumerate(subtask_files):
        with h5py.File(fpath, "r") as f:
            g = f["subtasks"]
            for k in g.keys():
                if not k.startswith("demo_subtasks_seq_"):
                    continue
                for start, end, label in g[k][()]:
                    if 0 <= label < K:
                        counts[ti, label] += int(end - start)

    fracs = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(subtask_files) + 2))
    bottom = np.zeros(len(subtask_files))
    y = np.arange(len(subtask_files))
    for c in range(K):
        ax.barh(y, fracs[:, c], left=bottom,
                color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                edgecolor="white", label=f"skill {c}")
        bottom += fracs[:, c]
    ax.set_yticks(y)
    ax.set_yticklabels([task_short_name(p) for p in subtask_files], fontsize=10)
    ax.set_xlabel("fraction of demonstration time")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(loc="lower right", ncol=K, fontsize=9)
    ax.set_title(f"Skill usage by task (K={K})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] skill-usage → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="/home/iw/lotus_test/lotus/skill_learning/results/dinov2_libero_object_image_only")
    ap.add_argument("--out-dir", default="/home/iw/lotus_test/lotus/skill_learning/results/cluster_viz")
    ap.add_argument("--dataset-category", default="libero_object",
                    help="subdir under skill_data/ to search (e.g. libero_object, libero_goal)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    subtask_files = collect_subtask_hdf5(args.exp_dir, args.dataset_category)
    if not subtask_files:
        raise SystemExit(f"no subtask hdf5 found under {args.exp_dir}/skill_data/{args.dataset_category}")

    K = parse_K_from_filename(subtask_files[0])
    print(f"exp_dir={args.exp_dir}  n_tasks={len(subtask_files)}  K={K}")

    render_timeline_grid(subtask_files, os.path.join(args.out_dir, "timeline_grid.png"), K)
    render_aggregate_skill_distribution(subtask_files, os.path.join(args.out_dir, "skill_usage.png"), K)
    render_tsne(
        os.path.join(args.exp_dir, "skill_data", "saved_feature_data.hdf5"),
        subtask_files,
        os.path.join(args.out_dir, "tsne_embeddings.png"),
    )


if __name__ == "__main__":
    main()
