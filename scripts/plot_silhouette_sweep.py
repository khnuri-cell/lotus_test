"""Plot silhouette scores from the LOTUS sweep log."""
import re
import sys
import matplotlib.pyplot as plt

log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/silhouette.log"
out_path = sys.argv[2] if len(sys.argv) > 2 else "/home/iw/lotus_test/lotus/skill_learning/results/cluster_viz/silhouette_sweep.png"
title_suffix = sys.argv[3] if len(sys.argv) > 3 else "LIBERO-OBJECT (6 tasks)"

pat = re.compile(r"\[silhouette\]\s+K=\s*(\d+)\s+score=([+-][\d.]+)")
ks, scores = [], []
selected = None
with open(log_path) as f:
    for line in f:
        m = pat.search(line)
        if m:
            ks.append(int(m.group(1))); scores.append(float(m.group(2)))
        m2 = re.search(r"selected K\* = (\d+)\s+\(max score = ([+-][\d.]+)\)", line)
        if m2:
            selected = (int(m2.group(1)), float(m2.group(2)))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ks, scores, marker="o", linewidth=2, color="#4363d8")
if selected is not None:
    ax.scatter([selected[0]], [selected[1]], color="#e6194B", s=180, zorder=5,
               label=f"selected K*={selected[0]} (silhouette={selected[1]:+.4f})")
    ax.axvline(selected[0], linestyle="--", color="#e6194B", alpha=0.4)
ax.set_xticks(ks)
ax.set_xlabel("Number of clusters K")
ax.set_ylabel("Mean silhouette score")
ax.set_title(f"LOTUS base-stage K selection — silhouette sweep on {title_suffix}")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
ax.axhline(0, color="black", alpha=0.2, linewidth=0.8)
plt.tight_layout()
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"saved: {out_path}")
