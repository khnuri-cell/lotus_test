#!/usr/bin/env bash
# End-to-end runbook: RoboCasa365 target/atomic-seen 18 tasks
# → LIBERO hdf5 → DINOv2 → hierarchical agglomeration → spectral clustering
# (silhouette sweep) → 4 visualizations.
#
# Run from repo root (/home/iw/lotus_test or /home/iw/code/lotus_test).
# Each stage is idempotent enough; comment out completed stages to resume.
#
# ⚠️  Full scale (all ~500 demos/task) = ~9100 demos / 2.23M frames.
#     Disk: ~30 GB+ for the converted hdf5s, ~20-40 GB for embedding hdf5s.
#     GPU: DINOv2 stage ~5-8 h on a single RTX 4070 Ti.
#     For a fast prototype set MAX_DEMOS=25 (~30 min total end-to-end).

set -euo pipefail

# ──────── Knobs ────────
MAX_DEMOS=${MAX_DEMOS:-25}             # 0 = use all demos per task
IMAGE_SIZE=${IMAGE_SIZE:-128}
EXP_NAME=${EXP_NAME:-dinov2_robocasa365_atomic_seen_image_only}
MODALITY_STR=${MODALITY_STR:-dinov2_agentview_eye_in_hand}
FEATURE_DIM=${FEATURE_DIM:-1536}
DATASET_CATEGORY=robocasa365_atomic_seen
SRC_ROOT=${SRC_ROOT:-/home/iw/code/robocasa/datasets/v1.0/target/atomic}
RAW_DATASET_DIR=datasets/${DATASET_CATEGORY}
RESULTS_ROOT=lotus/skill_learning/results/${EXP_NAME}
VIZ_OUT_DIR=lotus/skill_learning/results/cluster_viz_robocasa365_atomic_seen
K_SEARCH_LOW=${K_SEARCH_LOW:-2}
K_SEARCH_HIGH=${K_SEARCH_HIGH:-20}
N_CENTROID_FRAMES=${N_CENTROID_FRAMES:-5}

mkdir -p "${VIZ_OUT_DIR}"

# ──────── 1) Convert LeRobot → LIBERO hdf5 (one file per task) ────────
echo "[1/5] convert  src=${SRC_ROOT}  out=${RAW_DATASET_DIR}  max_demos=${MAX_DEMOS}"
python scripts/convert_robocasa365_atomic_seen.py \
    --src-root "${SRC_ROOT}" \
    --out-dir  "${RAW_DATASET_DIR}" \
    --image-size "${IMAGE_SIZE}" \
    --max-demos-per-task "${MAX_DEMOS}" \
    --skip-existing

# ──────── 2) DINOv2 representation per task ────────
# dinov2_repr.py is hard-coded to run from lotus/skill_learning/ and writes to
# results/{exp_name}/repr/{category}/{task}/embedding_{modality}_{dim}.hdf5
echo "[2/5] DINOv2  exp=${EXP_NAME}  modality=${MODALITY_STR}  dim=${FEATURE_DIM}"
( cd lotus/skill_learning && \
  python multisensory_repr/dinov2_repr.py \
    --exp-name      "${EXP_NAME}" \
    --modality-str  "${MODALITY_STR}" \
    --feature-dim   "${FEATURE_DIM}" \
    --dataset-category "${DATASET_CATEGORY}" )

# ──────── 3) Hierarchical agglomeration (per-demo bottom-up tree) ────────
echo "[3/5] hierarchical agglomeration"
( cd lotus/skill_learning && \
  python skill_discovery/hierarchical_agglomoration.py \
    exp_name="${EXP_NAME}" \
    modality_str="${MODALITY_STR}" \
    repr.z_dim="${FEATURE_DIM}" \
    agglomoration.dist=cos \
    agglomoration.footprint=global_pooling \
    +skip_tree_viz=true )    # 9000+ PNGs at full scale — keep off

# ──────── 4) Spectral clustering with silhouette sweep ────────
# K=-1 triggers the sweep in agglomoration_script.py (the K1 selection from
# paper Sec IV-A). The selected K* + per-K scores are echoed in stdout.
echo "[4/5] spectral clustering + silhouette sweep K∈[${K_SEARCH_LOW},${K_SEARCH_HIGH})"
SILHOUETTE_LOG=$(mktemp /tmp/silhouette_${DATASET_CATEGORY}.XXXXXX.log)
( cd lotus/skill_learning && \
  python skill_discovery/agglomoration_script.py \
    exp_name="${EXP_NAME}" \
    modality_str="${MODALITY_STR}" \
    repr.z_dim="${FEATURE_DIM}" \
    agglomoration.segment_scale=1 \
    agglomoration.min_len_thresh=30 \
    agglomoration.K=-1 \
    agglomoration.scale=0.01 \
    agglomoration.dist=cos \
    +agglomoration.k_search_low="${K_SEARCH_LOW}" \
    +agglomoration.k_search_high="${K_SEARCH_HIGH}" \
    +dataset_category="${DATASET_CATEGORY}" ) | tee "${SILHOUETTE_LOG}"

# ──────── 5) Visualizations ────────
echo "[5/5] visualizations → ${VIZ_OUT_DIR}"

# 5a) silhouette sweep curve
python scripts/plot_silhouette_sweep.py \
    "${SILHOUETTE_LOG}" \
    "${VIZ_OUT_DIR}/silhouette_sweep.png" \
    "RoboCasa365 atomic-seen 18"

# 5b) timeline grid + skill-usage bars + t-SNE (cluster & task colored)
python scripts/visualize_clustering.py \
    --exp-dir "${RESULTS_ROOT}" \
    --out-dir "${VIZ_OUT_DIR}" \
    --dataset-category "${DATASET_CATEGORY}"

# 5c) cluster centroid frames (per-cluster representative motion preview)
python scripts/visualize_cluster_centroids.py \
    --exp-dir "${RESULTS_ROOT}" \
    --raw-dataset-dir "${RAW_DATASET_DIR}" \
    --dataset-category "${DATASET_CATEGORY}" \
    --out-dir "${VIZ_OUT_DIR}" \
    --n-per-cluster "${N_CENTROID_FRAMES}" \
    --image-size "${IMAGE_SIZE}"

echo "done. outputs:"
ls -lh "${VIZ_OUT_DIR}"
