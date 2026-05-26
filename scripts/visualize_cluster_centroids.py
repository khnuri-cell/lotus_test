"""Render per-cluster representative frame grids from a LOTUS skill_data
spectral-clustering output.

For each of the K skill clusters, picks N "central" segments (closest to the
cluster centroid in the embedding space loaded from saved_feature_data.hdf5)
and draws their first / mid / last frames in a row.

Layout:
    rows = K clusters
    cols = N representative segments × 3 frames each (first|mid|last)

Usage:
    python scripts/visualize_cluster_centroids.py \\
        --exp-dir lotus/skill_learning/results/dinov2_robocasa365_atomic_seen_image_only \\
        --raw-dataset-dir datasets/robocasa365_atomic_seen \\
        --dataset-category robocasa365_atomic_seen \\
        --out-dir lotus/skill_learning/results/cluster_viz_robocasa365_atomic_seen \\
        --n-per-cluster 5
"""

import argparse
import glob
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np


def collect_subtask_hdf5(exp_dir, category):
    return sorted(glob.glob(os.path.join(exp_dir, "skill_data", category, "*.hdf5")))


def parse_K_from_filename(path):
    m = re.search(r"_K(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def task_name_from_subtask_path(path):
    base = os.path.basename(path)
    return re.split(r"_subtasks_", base, maxsplit=1)[0]


def load_raw_frames(raw_dataset_dir, category, task, ep_idx, start, end, image_size):
    """Return (first, mid, last) HWC uint8 frames for a (task, ep, start..end)."""
    h5_path = os.path.join(raw_dataset_dir, f"{task}.hdf5")
    with h5py.File(h5_path, "r") as f:
        imgs = f[f"data/demo_{ep_idx}/obs/agentview_rgb"]
        T = imgs.shape[0]
        s = max(0, min(int(start), T - 1))
        e = max(s, min(int(end), T - 1))
        m = (s + e) // 2
        frames = [imgs[s], imgs[m], imgs[e]]
    out = []
    for fr in frames:
        if fr.ndim == 3 and fr.shape[-1] != 3 and fr.shape[0] == 3:
            fr = np.transpose(fr, (1, 2, 0))
        if image_size and (fr.shape[0] != image_size or fr.shape[1] != image_size):
            try:
                import cv2
                fr = cv2.resize(fr, (image_size, image_size), interpolation=cv2.INTER_AREA)
            except ImportError:
                pass
        out.append(fr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True,
                    help="results/{exp_name} dir containing skill_data/ and skill_data/saved_feature_data.hdf5")
    ap.add_argument("--raw-dataset-dir", required=True,
                    help="root holding the LIBERO-style hdf5s, e.g. datasets/robocasa365_atomic_seen")
    ap.add_argument("--dataset-category", required=True,
                    help="subdir under skill_data/ (e.g. robocasa365_atomic_seen)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-cluster", type=int, default=5)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--feature-file", default="saved_feature_data.hdf5",
                    help="Inside {exp_dir}/skill_data/. Should hold embeddings+cluster_labels+task_ids+locs.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    subtask_files = collect_subtask_hdf5(args.exp_dir, args.dataset_category)
    if not subtask_files:
        raise SystemExit(f"no subtask hdf5 under {args.exp_dir}/skill_data/{args.dataset_category}")
    K = parse_K_from_filename(subtask_files[0])
    task_names = [task_name_from_subtask_path(p) for p in subtask_files]
    print(f"exp_dir={args.exp_dir}  n_tasks={len(task_names)}  K={K}")

    # 1) Load segment-level features from saved_feature_data.hdf5 (written by
    #    agglomoration_script). Required: embeddings (N,D), cluster_labels (N,),
    #    task_ids (N,) and a way to map each segment back to (task, ep, start, end).
    feat_path = os.path.join(args.exp_dir, "skill_data", args.feature_file)
    if not os.path.exists(feat_path):
        raise SystemExit(
            f"missing {feat_path} — re-run agglomoration_script with saved features dumped"
        )
    with h5py.File(feat_path, "r") as f:
        emb = f["embeddings"][()]
        cluster_labels = f["cluster_labels"][()]
        task_ids = f["task_ids"][()]
        demo_indices = f["demo_indices"][()]
        seg_start = f["seg_start"][()]
        seg_end = f["seg_end"][()]
    print(f"  loaded {len(emb)} segments, dim={emb.shape[1]}, K_unique={len(np.unique(cluster_labels))}")

    # 2) For each cluster, pick N segments closest to the cluster centroid
    fig_w, fig_h = 1.6, 1.6
    cols = args.n_per_cluster * 3
    fig, axes = plt.subplots(K, cols, figsize=(fig_w * cols, fig_h * K), squeeze=False)

    for c in range(K):
        mask = cluster_labels == c
        if mask.sum() == 0:
            for j in range(cols):
                axes[c, j].axis("off")
            continue
        emb_c = emb[mask]
        idx_c = np.where(mask)[0]
        centroid = emb_c.mean(axis=0)
        dists = np.linalg.norm(emb_c - centroid, axis=1)
        order = np.argsort(dists)[: args.n_per_cluster]
        picked = idx_c[order]

        for i, seg_idx in enumerate(picked):
            tid = int(task_ids[seg_idx])
            ep = int(demo_indices[seg_idx])
            s = int(seg_start[seg_idx])
            e = int(seg_end[seg_idx])
            task = task_names[tid] if tid < len(task_names) else f"task{tid}"
            try:
                frames = load_raw_frames(args.raw_dataset_dir, args.dataset_category,
                                          task, ep, s, e, args.image_size)
            except Exception as ex:
                frames = [np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)] * 3
                print(f"  [skill {c}] failed {task} ep{ep} {s}-{e}: {ex}")

            for k, fr in enumerate(frames):
                ax = axes[c, i * 3 + k]
                ax.imshow(fr)
                ax.set_xticks([]); ax.set_yticks([])
                if k == 0:
                    ax.set_ylabel(f"skill {c}" if i == 0 else "", fontsize=11, rotation=0,
                                  labelpad=28, ha="right", va="center", fontweight="bold")
                ax.set_title(["first", "mid", "last"][k] +
                             (f" — {task[:14]} ep{ep}" if k == 1 else ""),
                             fontsize=7)

    fig.suptitle(f"Cluster centroid frames (K={K}, top-{args.n_per_cluster} closest-to-centroid)",
                 fontsize=13)
    plt.tight_layout(rect=[0.02, 0, 1, 0.97])
    out = os.path.join(args.out_dir, "cluster_centroid_frames.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] centroid frames → {out}")


if __name__ == "__main__":
    main()
