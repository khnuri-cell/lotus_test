# LOTUS Fork — Handover (2026-05-23)

새 서버로 옮긴 직후 빠르게 실험을 이어가기 위한 인수인계 문서. 한 번
훑어보면 "데이터 어디서 받고 / 환경 어떻게 띄우고 / 무슨 명령부터
때리면 되는지" 가 잡히는 게 목표.

> 작성 시점 environment: Linux 6.17, CUDA 11.8 + RTX 4070 Ti, Python 3.9.19,
> Docker. 새 서버 사양이 다르면 §환경 섹션의 핀 버전 일부 조정 필요.

---

## 0. 프로젝트 한 줄 요약

LOTUS (UT-Austin RPL, ICRA 2024) 의 unsupervised skill discovery 파이프라인 fork.
**원 paper 와 달리** LIBERO + RoboCasa 두 도메인에서 같은 파이프라인을 돌리고,
silhouette-기반 K 자동 선택 / 시각화 도구가 추가돼 있다.

- Paper 가 다룬 데이터셋: LIBERO-OBJECT, LIBERO-GOAL, LIBERO-50, MUTEX (real)
- 이 fork 가 추가로 다루는 데이터셋:
  - **DAVIAN-Robotics/robocasa-H50** (1293 demo, 24 atomic skill family)
  - **RoboCasa365 v1.0 target/atomic-seen** (18 task, ~9100 demo, ~2.23M frame)

---

## 1. 새 서버 1회성 셋업

### 1.1 코드 받기
```bash
git clone https://github.com/khnuri-cell/lotus_test.git
cd lotus_test
git remote add upstream https://github.com/UT-Austin-RPL/Lotus.git   # 선택
```

### 1.2 Docker 빌드 + 실행
```bash
docker compose build   # docker/dockerfile 사용. ~10분.
docker compose up -d
docker exec -it lotus_test bash
# 컨테이너 안에서:
cd ~/lotus_test
pip install -e .
# LIBERO 환경 따로 설치 (vendor 안 했음)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO ~/LIBERO
cd ~/LIBERO && pip install -e .
```

⚠️ `docker/dockerfile` 의 핀들은 RTX 4070 Ti (Ada, sm_89) 기준. 다른 GPU 면
`torch==2.0.1+cu118` → `cu121` 또는 cu118 호환 다른 버전으로 조정.

### 1.3 데이터 받기

#### A. LIBERO (paper 재현용)
```bash
python libero_benchmark_scripts/download_libero_datasets.py --datasets libero_object
python libero_benchmark_scripts/download_libero_datasets.py --datasets libero_goal
# LIBERO-50 (kitchen) 이 필요하면 libero_100 받고 kitchen subset 만 사용
```
→ `datasets/libero_object/*.hdf5`, `datasets/libero_goal/*.hdf5` 생성.

#### B. RoboCasa-H50 (1293 demo, 24 atomic 합본)
**원본 LeRobot:**
```bash
huggingface-cli download DAVIAN-Robotics/robocasa-H50 \
    --repo-type dataset \
    --local-dir ~/.cache/huggingface/lerobot/DAVIAN-Robotics/robocasa-H50
```
**LIBERO 형식 변환** (`scripts/lerobot_to_libero.py` 사용):
```bash
# all 1293 demos
python scripts/lerobot_to_libero.py \
    --repo-id DAVIAN-Robotics/robocasa-H50 \
    --output datasets/robocasa_h50_all/robocasa_h50_all.hdf5 \
    --task-name robocasa_h50 \
    --image-size 128

# 50 demo 샘플
python scripts/lerobot_to_libero.py \
    --repo-id DAVIAN-Robotics/robocasa-H50 \
    --output datasets/robocasa_h50_50ep/robocasa_h50_50ep.hdf5 \
    --task-name robocasa_h50 \
    --image-size 128 \
    --max-episodes 50
```

#### C. RoboCasa365 target/atomic-seen 18 task (★ 현재 진행 중 실험)
원본은 LeRobot v2.1 포맷, 18개 task 각각 `~500 demo` (총 ~9100 demo). 원본
위치는 [`robocasa` 리포의 `datasets/v1.0/target/atomic/`](https://robocasa.ai/docs/datasets/using_datasets.html).
이전 서버에서는 `/home/iw/code/robocasa/datasets/v1.0/target/atomic/` 에 있었음.

```bash
# 새 서버에서 robocasa 리포 받아 download_datasets.py 실행
git clone https://github.com/robocasa/robocasa ~/code/robocasa
cd ~/code/robocasa
python robocasa/scripts/download_datasets.py --task_set atomic_seen --dataset_soup target_atomic_seen
# → datasets/v1.0/target/atomic/{TaskName}/{date}/lerobot/...
```

변환 (이미 작성된 일괄 변환 스크립트 사용):
```bash
cd ~/lotus_test
# 빠른 프로토타입 (task 당 25 demo, ~3GB)
MAX_DEMOS=25 bash scripts/run_robocasa365_atomic_seen.sh
# 풀스케일 (task 당 전체 ~500 demo, ~30GB+, ~5-8h GPU)
MAX_DEMOS=0  bash scripts/run_robocasa365_atomic_seen.sh
```
runbook (`scripts/run_robocasa365_atomic_seen.sh`) 안에서 `SRC_ROOT` ENV 로
원본 데이터 위치 조정 가능.

---

## 2. 핵심 파이프라인 (skill discovery)

LIBERO-style hdf5 (`datasets/{category}/{task}.hdf5`) 가 준비됐다고 가정.

```bash
cd lotus/skill_learning

# 1) DINOv2 표현 추출 → results/{exp}/repr/{category}/{task}/embedding_*.hdf5
python multisensory_repr/dinov2_repr.py \
    --exp-name dinov2_libero_object_image_only \
    --modality-str dinov2_agentview_eye_in_hand \
    --feature-dim 1536 \
    --dataset-category libero_object         # ★ 추가된 옵션

# 2) per-demo hierarchical agglomerative tree
python skill_discovery/hierarchical_agglomoration.py \
    exp_name=dinov2_libero_object_image_only \
    modality_str=dinov2_agentview_eye_in_hand \
    repr.z_dim=1536 \
    agglomoration.dist=cos \
    agglomoration.footprint=global_pooling \
    +skip_tree_viz=true                       # ★ 풀스케일에서 PNG 폭발 방지

# 3) spectral clustering (silhouette sweep)
python skill_discovery/agglomoration_script.py \
    exp_name=dinov2_libero_object_image_only \
    modality_str=dinov2_agentview_eye_in_hand \
    repr.z_dim=1536 \
    agglomoration.segment_scale=1 \
    agglomoration.min_len_thresh=30 \
    agglomoration.K=-1 \                       # ★ -1 → silhouette 자동 선택
    +agglomoration.k_search_low=2 \
    +agglomoration.k_search_high=20 \
    agglomoration.scale=0.01 \
    agglomoration.dist=cos \
    +dataset_category=libero_object            # ★
```

### dataset registry 가 등록된 곳 (3 파일)

새 카테고리 추가시 **세 군데 모두 수정** 필요:

| 파일 | 무엇 |
|---|---|
| `lotus/skill_learning/multisensory_repr/dinov2_repr.py` | `DATASET_TASKS` dict |
| `lotus/skill_learning/skill_discovery/agglomoration_script.py` | `DATASET_TASKS` dict |
| `lotus/skill_learning/skill_discovery/hierarchical_agglomoration.py` | 카테고리 allowlist 1줄 |

현재 등록된 카테고리: `libero_object`, `libero_goal`, `robocasa_h50_50ep`,
`robocasa_h50_all`, `robocasa365_atomic_seen` (18 task).

---

## 3. 시각화 도구

전부 `scripts/` 하위. clustering 결과 (`results/{exp}/skill_data/`) 를 입력으로.

| 스크립트 | 결과물 |
|---|---|
| `scripts/plot_silhouette_sweep.py LOGFILE OUTPATH "TITLE"` | K vs silhouette 곡선 + K\* 강조 |
| `scripts/visualize_clustering.py --exp-dir ... --out-dir ... --dataset-category ...` | timeline_grid.png, skill_usage.png, tsne_embeddings.png (cluster × task) |
| `scripts/visualize_cluster_centroids.py --exp-dir ... --raw-dataset-dir ... --dataset-category ... --out-dir ... --n-per-cluster 5` | cluster_centroid_frames.png (K rows × N segments × 3 frames) |

end-to-end runbook 은 `scripts/run_robocasa365_atomic_seen.sh` 참조.

기존 산출물 (이전 서버에서 생성, 백업 안 함):
- `lotus/skill_learning/results/cluster_viz/` (libero_object)
- `lotus/skill_learning/results/cluster_viz_goal/` (libero_goal)
- `lotus/skill_learning/results/cluster_viz_robocasa/` (h50_50ep)
- `lotus/skill_learning/results/cluster_viz_robocasa_all/` (h50_all)

→ 이 PNG 들은 `results/` 디렉토리째 gitignored 라 리포에 없음. **새 서버에서는
재생성 필요**. (필요하면 옮기기 전에 별도로 scp 백업.)

---

## 4. 진행 중인 작업 / 다음 할 일

### 직전까지 한 일 (2026-05-23 기준)
- ✅ Paper 가 어떤 데이터셋을 썼는지 확인 → LIBERO 3 suite + MUTEX. **RoboCasa 안 씀**.
- ✅ DAVIAN-Robotics/robocasa-H50 = 24 atomic family / 1293 demo / 242 unique
      instruction. → `robocasa_h50_all_tasks.tsv` (root)
- ✅ RoboCasa365 target/atomic-seen 18 task 변환·시각화 파이프라인 완성
      (`scripts/convert_robocasa365_atomic_seen.py`, `scripts/run_robocasa365_atomic_seen.sh`,
      `scripts/visualize_cluster_centroids.py`)
- ✅ silhouette sweep K 자동 선택 활성화 (`agglomoration.K=-1`)
- ✅ 3 dataset registry 에 `robocasa365_atomic_seen` 등록

### 새 서버에서 바로 할 일
1. **데이터 받기** — §1.3 C (RoboCasa365 atomic-seen 18 LeRobot 원본).
2. **`MAX_DEMOS=25 bash scripts/run_robocasa365_atomic_seen.sh`** 로 sanity-check
   (~30분 GPU). silhouette sweep 곡선 + t-SNE + timeline + centroid frames
   생성되는지 확인.
3. 결과 합리적이면 `MAX_DEMOS=0` 로 풀스케일 실행 (~5-8h).
4. K\* 가 너무 작거나 크면 `K_SEARCH_LOW/HIGH`, `agglomoration.scale` (현재 0.01),
   `agglomoration.min_len_thresh` (30) 조정.

### 가능성 있는 후속 실험
- `robocasa365_atomic_seen` 의 18 task 를 LOTUS lifelong split 으로 나눠
  (e.g. base 12 + lifelong 6×1) 정책 학습까지 진행 — paper Table I 의
  RoboCasa 버전 확장.
- per-demo task instruction (변환시 `data/demo_{i}.attrs["task_instruction"]`
  로 저장됨) 을 활용해 language 별 segmentation 비교.
- DINOv2 vs DINOv3 ablation (이전 서버에 dinov3_test 컨테이너 있음).

---

## 5. 알려진 함정 / 디버깅 메모

| 증상 | 원인 / 대응 |
|---|---|
| `ImportError: dinov2.models` | cwd 가 `lotus/skill_learning/` 가 아닐 때. `dinov2_repr.py` 의 `sys.path.insert` 가 `__file__` 기준이지만, 원래 `sys.path.append('dinov2')` 라서 cwd 의존이었다. 항상 `cd lotus/skill_learning` 후 실행. |
| `cmake / egl_probe` 빌드 실패 (dockerfile) | `RUN python3.9 -m pip install cmake==3.27.7 ... && pip install --no-build-isolation egl_probe` 블록 유지할 것. PEP 517 격리 환경에서 pip 의 cmake wrapper 가 자기 모듈을 못 찾는 이슈. |
| `hierarchical_agglomoration.py` 가 9000+ PNG 를 그려서 디스크 폭주 | `+skip_tree_viz=true` (이 fork 가 추가한 hydra override). |
| robocasa LeRobot 변환 시 `joint_states` 가 0 | LeRobot 메타에 7-DOF joint 없음. image-only skill discovery 에는 무해. policy 학습으로 가면 proprio modality 끌 것. |
| converted hdf5 가 single language instruction 만 가짐 | `lerobot_to_libero.py` 는 `data.attrs.problem_info` 에 단일 `--task-name` 만 저장. per-demo 라벨은 `convert_robocasa365_atomic_seen.py` 가 `demo_{i}.attrs["task_instruction"]` 로 따로 박아줌. |
| dinov2 가 hydra config 를 cwd 에 덤프 (`config.yaml`) | gitignore 처리됨. |
| RoboCasa365 LeRobot 은 v2.1 (`episode_*.parquet`), robocasa-H50 은 v3.0 (`file-*.parquet`) | `LeRobotDataset(root=...)` 둘 다 자동 처리. 단 `ds.meta.episodes[i]` 접근 형태가 약간 다를 수 있어 `convert_robocasa365_atomic_seen.py` 에 try/except 들어가 있음. |

---

## 6. Claude Code 대화 이어가기

**이 대화는 새 서버로 이어지지 않습니다.** Claude Code 의 conversation history
는 호스트의 `~/.claude/projects/-home-iw-code-lotus-test/` (CWD-기반 해시 경로)
에 로컬로만 저장되고, 서버 간 동기화되지 않음.

새 서버에서 Claude Code 를 띄우면:
1. `cd ~/lotus_test && claude` (또는 `claude code`) 로 시작.
2. 첫 메시지에서 **"이 HANDOVER.md 읽고 현재 상황 파악해줘"** 라고 주면, 새
   conversation 이 이 문서 + 코드 상태로 컨텍스트를 잡는다.
3. 메모리(user/feedback/project) 도 호스트 로컬이라 이어지지 않음. 이 fork
   관련 핵심 메모리는 본 문서의 §0 / §2 / §5 에 녹여 뒀음. 새 서버에서 다시
   짧게 알려주면 자동 메모리 시스템이 새로 저장한다.

이전 서버의 메모리 백업이 필요하면 `~/.claude/projects/-home-iw-code-lotus-test/memory/`
디렉토리 통째로 scp 후 새 서버의 같은 경로에 복원하면 user/project/reference
메모리가 살아남는다. (단 새 서버의 절대경로가 다르면 디렉토리 이름의
해시 부분이 달라져 자동 로드되지 않을 수 있음.)

---

## 7. 연락처 / 출처

- 본인: khnuri@hanyang.ac.kr (Hanyang Univ.)
- 원 LOTUS: https://ut-austin-rpl.github.io/Lotus/  ·  paper https://arxiv.org/pdf/2311.02058
- RoboCasa365: https://robocasa.ai/  ·  github.com/robocasa/robocasa
- DAVIAN-Robotics/robocasa-H50: https://huggingface.co/datasets/DAVIAN-Robotics/robocasa-H50
