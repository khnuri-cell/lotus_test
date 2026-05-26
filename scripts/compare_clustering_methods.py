#!/usr/bin/env python3
"""Compare Tier-A skill-discovery clustering methods on the SAME LOTUS embeddings.

We hold the DINOv2 segment embeddings + segments fixed (exactly what LOTUS / the
viewer used) and swap ONLY the cluster-assignment step, at matched K:

  LOTUS    : saved spectral-clustering labels (the pipeline's own result)
  BUDS     : bottom-up agglomerative clustering (Ward)            [its signature]
  XSkill   : SwAV-style Sinkhorn-Knopp prototype clustering        [balanced, cosine]
  CompILE  : Gaussian Mixture (categorical latent-code analog)

For each method we report:
  intrinsic (no labels) : silhouette(cosine), Davies-Bouldin, Calinski-Harabasz
  extrinsic vs task_ids : NMI, ARI, homogeneity, completeness   (task = proxy GT)

and render a per-dataset t-SNE figure colored by each method + by task.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, normalized_mutual_info_score,
                             adjusted_rand_score, homogeneity_score, completeness_score)

ROOT = "/home/iw/lotus_test/lotus/skill_learning/results"
RUNS = {
    "libero_object": f"{ROOT}/dinov2_libero_object_image_only_10",
    "libero_goal":   f"{ROOT}/dinov2_libero_goal_image_only_10",
    "robocasa365":   f"{ROOT}/dinov2_robocasa365_atomic_seen_image_only_18",
}
OUT = Path(f"{ROOT}/method_comparison")
OUT.mkdir(parents=True, exist_ok=True)


def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def sinkhorn(logits, n_iter=3):
    Q = np.exp(logits - logits.max())
    Q /= Q.sum()
    N, K = Q.shape
    for _ in range(n_iter):
        Q /= Q.sum(0, keepdims=True); Q /= K
        Q /= Q.sum(1, keepdims=True); Q /= N
    return Q * N


def xskill_sinkhorn(X, K, n_iter=50, eps=0.05, seed=0):
    """SwAV/XSkill-style: balanced prototype assignment via Sinkhorn-Knopp."""
    C = KMeans(K, n_init=3, random_state=seed).fit(X).cluster_centers_
    C = l2norm(C)
    assign = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        Q = sinkhorn(X @ C.T / eps)
        assign = Q.argmax(1)
        for k in range(K):
            m = assign == k
            if m.any():
                C[k] = X[m].mean(0)
        C = l2norm(C)
    return assign


def cluster_all(X, K, lotus_labels, seed=0):
    """Return dict method -> labels (all at K clusters)."""
    Xn = l2norm(X)
    out = {}
    out["LOTUS\n(spectral)"] = lotus_labels
    out["BUDS\n(agglom.)"] = AgglomerativeClustering(
        n_clusters=K, linkage="ward").fit_predict(Xn)
    out["XSkill\n(Sinkhorn)"] = xskill_sinkhorn(Xn, K, seed=seed)
    out["CompILE\n(GMM)"] = GaussianMixture(
        n_components=K, covariance_type="diag", reg_covar=1e-4,
        n_init=2, random_state=seed).fit(Xn).predict(Xn)
    return out


def metrics(X, labels, task_ids):
    Xn = l2norm(X)
    nuniq = len(np.unique(labels))
    sil = silhouette_score(Xn, labels, metric="cosine") if nuniq > 1 else float("nan")
    db = davies_bouldin_score(Xn, labels) if nuniq > 1 else float("nan")
    ch = calinski_harabasz_score(Xn, labels) if nuniq > 1 else float("nan")
    return {
        "n_clusters": nuniq,
        "silhouette_cos": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "NMI_vs_task": normalized_mutual_info_score(task_ids, labels),
        "ARI_vs_task": adjusted_rand_score(task_ids, labels),
        "homogeneity": homogeneity_score(task_ids, labels),
        "completeness": completeness_score(task_ids, labels),
    }


def main():
    rows = []
    for name, d in RUNS.items():
        with h5py.File(f"{d}/skill_data/saved_feature_data.hdf5", "r") as f:
            X = f["embeddings"][()].astype(np.float64)
            lotus = f["cluster_labels"][()].astype(int)
            task = f["task_ids"][()].astype(int)
        K = len(np.unique(lotus))
        print(f"\n=== {name}  (segs={len(X)}, dim={X.shape[1]}, K={K}, tasks={len(np.unique(task))}) ===")
        labelsets = cluster_all(X, K, lotus)

        # shared t-SNE
        print("  t-SNE...", flush=True)
        emb2d = TSNE(n_components=2, perplexity=30, init="pca",
                     random_state=0).fit_transform(l2norm(X))

        # metrics table
        print(f"  {'method':18} {'sil(cos)':>9} {'DB↓':>7} {'CH↑':>8} {'NMI':>6} {'ARI':>6} {'homog':>6} {'compl':>6}")
        for m, lab in labelsets.items():
            mt = metrics(X, lab, task)
            mt_row = {"dataset": name, "method": m.replace("\n", " "), **mt}
            rows.append(mt_row)
            print(f"  {m.replace(chr(10),' '):18} {mt['silhouette_cos']:9.3f} "
                  f"{mt['davies_bouldin']:7.2f} {mt['calinski_harabasz']:8.1f} "
                  f"{mt['NMI_vs_task']:6.3f} {mt['ARI_vs_task']:6.3f} "
                  f"{mt['homogeneity']:6.3f} {mt['completeness']:6.3f}")

        # figure: 1 row, 5 panels (4 methods + task coloring)
        panels = list(labelsets.items()) + [("by TASK", task)]
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.2))
        for ax, (title, lab) in zip(axes, panels):
            ax.scatter(emb2d[:, 0], emb2d[:, 1], c=lab, cmap="tab20", s=8, alpha=0.8, linewidths=0)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{name}  —  t-SNE of {len(X)} segments  (K={K}, tasks={len(np.unique(task))})",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_png = OUT / f"{name}_tsne_methods.png"
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        print(f"  ✅ {out_png}")

    # write CSV
    csv_path = OUT / "metrics.csv"
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ metrics -> {csv_path}")
    print("done.")


if __name__ == "__main__":
    main()
