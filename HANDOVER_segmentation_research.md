# Handover — Trajectory Segmentation Method Comparison (research)

> 이 문서는 새 대화로 작업을 이어받기 위한 인수인계 노트입니다.
> 작성 시점 기준 상태를 담았고, 코드/데이터 경로는 새 대화에서 다시 검증하세요.

---

## 0. 한 줄 요약

LOTUS fork 에서, **trajectory 를 "어디서·어떻게 끊는가(segmentation criterion/model)"** 를
여러 skill-discovery 방법 간에 **시각적 + 수치적으로 비교**하는 연구를 준비 중.
현재까지: 후보 논문 4개 선정 → 클러스터링 ablation 실험 1회(축이 틀렸음을 확인) →
**segmentation 비교로 재정렬** → 현재 데이터가 atomic 이라 long-horizon 데이터 확보가 필요하다는 결론.

---

## 1. 환경 / 인프라

- **repo**: `/home/iw/code/lotus_test`  (branch: `nuri`, main: `master`)
- **컨테이너**: `lotus_test` (docker). 내부 repo 경로 = `/home/iw/lotus_test`
  - 실행 예: `docker exec -i lotus_test python3 /home/iw/lotus_test/scripts/<script>.py`
  - **`-i` 필수** (heredoc/stdin 전달용). 없으면 stdin 안 들어감.
  - 패키지: sklearn 1.6.1, h5py 3.14.0, scipy, matplotlib 3.5.3, numpy. (호스트 python3 에는 없음 → 항상 컨테이너에서)
  - 또 다른 이미지 `dinov3_test:latest` (Python 3.10 + lerobot) — 데이터 변환용
- **GPU**: RTX 4070 Ti (sm_89), torch 2.0.1+cu118
- **주의**: Claude 의 `git push` 는 하네스 보안정책상 **하드 블록**됨(public repo data exfiltration). push 는 사용자가 `!git ...` 로 직접 실행해야 함.

---

## 2. 데이터셋 (로컬 `datasets/`)

| dataset | task 종류 | demos | 성격 (robocasa 기준) |
|---|---|---:|---|
| `libero_object` | 10 (pick-place, 물체만 다름) | 500 | **atomic** (short-horizon) |
| `libero_goal`   | 10 (목표만 다름) | 500 | **atomic** |
| `robocasa365_atomic_seen` | 18 atomic kitchen | 450 | **atomic** |
| `robocasa_h50_50ep` / `robocasa_h50_all` | 합본 1파일 | - | reward 전부 0, instruction 없음 |

### 핵심 사실 (실측 확인됨)
- **원본 raw demo 는 안 잘려 있음** — task별 통짜 trajectory (예: TurnOffStove demo_0 = 177프레임 연속, segment 키 없음).
- LOTUS 가 사후에 segmentation 함. 그 결과 **demo 당 segment 가 ~2개** (robocasa atomic 은 1.57개, 절반은 통째로).
  → atomic 데이터는 trajectory 당 경계가 ~1개라서 **segmentation 방법 비교 무대로 부적합**.
- **ground-truth sub-task 경계 라벨이 어디에도 없음** → boundary-F1 같은 정량평가 불가.

### raw demo 스키마 (LIBERO-style HDF5)
```
data/demo_{i}/
  actions (T,12), dones, rewards(전부0), robot_states(T,16), states(T,16)
  obs/{agentview_rgb(T,128,128,3), eye_in_hand_rgb, ee_states, gripper_states, joint_states}
  attrs: task_instruction, num_samples, model_file
```

### LOTUS 결과 스키마 (`results/<exp>/skill_data/saved_feature_data.hdf5`)
```
embeddings (N_seg, 4608)   # DINOv2 global_pooling concat (agentview+eye_in_hand)
cluster_labels (N_seg,)    # LOTUS spectral clustering 결과
task_ids (N_seg,)          # 각 segment 의 원 task
demo_indices, seg_start, seg_end (N_seg,)
```
- viewer/실험이 쓰는 canonical run:
  - libero_object → `results/dinov2_libero_object_image_only_10` (segs=996, K=3, 10 tasks)
  - libero_goal   → `results/dinov2_libero_goal_image_only_10`   (segs=993, K=15, 10 tasks)
  - robocasa365   → `results/dinov2_robocasa365_atomic_seen_image_only_18` (segs=708, K=37, 18 tasks)

---

## 3. 시각화 viewer (이전 작업, 완료됨)

- **로컬**: `viewer/` (484MB, 풀화질 PNG). 서빙: `cd viewer && python3 -m http.server 8765`
- **공개 호스팅**: https://khnuri-cell.github.io/lotus-viewer/  (repo `khnuri-cell/lotus-viewer`, public)
  - 경량판 `viewer_web/` (115MB, PNG→JPG q82). `scripts/make_web_viewer.py` 로 생성.
  - `viewer_web/` 은 자체 git repo. 갱신: `make_web_viewer.py` → commit → 사용자가 `!git -C viewer_web push`
- `viewer/`, `viewer_web/` 둘 다 부모 `.gitignore` 에 제외.
- 보안: 작업 중 GitHub PAT 2개 + WandB key 가 채팅에 노출됨 → **사용자가 revoke 해야 함** (이 문서엔 미포함).

---

## 4. 연구 질문 / 후보 논문 (Tier A — "임베딩에서 trajectory 묶어 분류")

### 후보 4개 + 인용수 (Semantic Scholar 기준, Google Scholar 는 1.5~2배)
| 방법 | 학회/연도 | SS 인용 | 표현(embedding) | **끊는 기준(cutting criterion)** |
|---|---|---:|---|---|
| **LOTUS** (베이스) | ICRA 2024 | 70 | Frozen DINOv2 (시각) | DINO 특징 change-point (agglomerative tree cut) |
| **BUDS** | RA-L 2022 | 113 | vision **+ proprioception** | multimodal change-point (bottom-up tree) |
| **XSkill** | CoRL 2023 | 114 | self-sup time-contrastive (사람+로봇) | skill **prototype 전환점** (Sinkhorn) |
| **CompILE** | ICML 2019 | ~200 | end-to-end latent (이미지X, toy) | **학습된 경계예측** (한 latent code 로 복원되는 구간) |
| ~~Motion2Vec~~ | ICRA 2020 | ~150 | Siamese metric learning | (제외됨) |

- 논문 링크: LOTUS arXiv:2311.02058 / BUDS arXiv:2109.13841 (github UT-Austin-RPL/BUDS) /
  XSkill arXiv:2307.09955 (github real-stanford/xskill) / CompILE arXiv:1812.01483 (github tkipf/compile)
- Tier B(skill discovery 전반, 클러스터링 아님): Relay(543), Play-LMP(488), SPiRL(298), OPAL(188) — 비교 대상에서 제외하기로.

### 끊는 철학 요약
- LOTUS=**본다(시각)** / BUDS=**보고+느낀다(시각+proprio)** / XSkill=**결과로 묶는다(action effect, embodiment 불변)** / CompILE=**행동을 압축한다(latent 재구성)**

---

## 5. 실험 1 (완료) — 클러스터링 ablation [축이 틀렸음, 참고용]

- 스크립트: `scripts/compare_clustering_methods.py`
- 내용: **segment 고정**(LOTUS 가 자른 것 재사용) + 임베딩 DINO 통일, **클러스터 할당 알고리즘만** 교체
  - LOTUS=saved spectral / BUDS=agglomerative(ward) / XSkill=Sinkhorn prototype / CompILE=GMM
- 지표: silhouette(cos), Davies-Bouldin, Calinski-Harabasz + task_ids 대비 NMI/ARI/homogeneity/completeness
- 산출: `results/method_comparison/{dataset}_tsne_methods.png` + `metrics.csv`

### 결과 요지
| dataset | 관찰 |
|---|---|
| libero_object (K=3) | 전반적으로 낮음. CompILE/XSkill silhouette 약간↑, BUDS NMI↑ |
| libero_goal (K=15) | 가장 잘 뭉침 (NMI 0.71~0.76). BUDS NMI/ARI 1등, CompILE silhouette 1등 |
| robocasa365 (K=37) | **LOTUS spectral silhouette 음수(-0.063)** ← 최약체. BUDS/XSkill/CompILE 은 양수 + NMI↑ |

### ⚠️ 이 실험의 validity 한계 (중요)
1. 각 논문의 핵심 기여(표현/segmentation)를 제거하고 클러스터링 함수만 바꾼 것 → **"4개 논문 비교"가 아님**, "LOTUS 파이프라인 클러스터링 ablation"일 뿐.
2. **task_ids 를 정답으로 한 NMI 는 논리적 결함**: skill 은 task 를 가로질러 재사용돼야 함 → task 와 1:1 매칭이 높은 게 오히려 나쁠 수 있음.
3. silhouette 은 구형 클러스터 선호 → k-means 계열(GMM/Sinkhorn)에 편향.
4. segment(경계)를 LOTUS 것으로 고정 → **segmentation 비교 자체가 불가능한 구조였음**.

---

## 6. 핵심 재정렬 — 진짜 하려는 것

사용자 실제 목표: **"trajectory 를 끊는 기준/모델"을 research 하고 성능차이를 시각+수치로 확인.**

→ 실험 1(클러스터링)은 곁가지였고, **segmentation(cutting) 비교가 본체**.
→ segmentation 을 비교하려면 **raw 통짜 trajectory 에서 출발해 각 방법이 독립적으로 cut** 해야 함
   (LOTUS 가 잘라놓은 걸 재사용하면 안 됨). 필요한 입력 = **per-frame DINO feature**.

### 합의된 프레이밍 (방어 가능 버전)
> **"표현(DINO)을 고정하고, segmentation criterion 만 바꿔가며 비교"**
> — 자르는 방식 자체의 효과를 분리. (선택적으로 각 방법 native 표현을 2번째 축으로 추가)

### 단계적 계획
| 단계 | 내용 | 비용 |
|---|---|---|
| 0 | long-horizon 데이터(LIBERO-LONG/libero_10 등) + per-frame DINO feature 인프라 확인 | 즉시 |
| 1 | 같은 raw trajectory 에 4개 cutting criterion 적용 → **타임라인 시각화** | 가벼움 |
| 2 | proxy 수치 (boundary sharpness, demo간 일관성, 방법간 경계 일치도) | 가벼움 |
| 3 | **downstream 성공률** (각 segmentation 으로 skill 학습 → task 성공) ← field 표준, 본진 | 무거움(GPU) |

### 평가지표 결정사항
- GT 경계 없음 → boundary-F1 불가.
- **downstream 성공률이 정직한 본진 지표** (LOTUS/BUDS/OPAL 다 이걸로 평가).
- proxy 지표는 보조. task-NMI 는 skill≠task 라 지양.

### 데이터 결정사항
- atomic(libero_object/goal, robocasa365) = trajectory 당 경계 ~1개 → 비교 무대로 부적합.
- **long-horizon 필요**: LIBERO-LONG(=LIBERO-10, composite) 또는 robocasa composite.
- LIBERO-GOAL/OBJECT 는 robocasa 기준 **atomic** 에 해당 (composite 아님). LIBERO-LONG 이 composite.

---

## 7. 즉시 다음 스텝 (새 대화에서 시작점)

1. **LIBERO-LONG(libero_10) 데이터 확보 가능 여부 확인** (LIBERO 공식 배포).
2. **per-frame DINO feature 추출 경로 확인** — 현재 saved_feature_data 는 segment 단위. raw → per-frame feature 파이프라인(`lotus/skill_learning/multisensory_repr/dinov2_repr.py`) 점검.
3. 확보되면: raw trajectory → 4개 cutting criterion(LOTUS/BUDS/XSkill/CompILE) 독립 적용 → 타임라인 시각화(단계 1) 부터.

---

## 8. 관련 파일 (이번 작업에서 생성/수정)

- `scripts/compare_clustering_methods.py` — 실험1 (클러스터링 ablation)
- `scripts/make_web_viewer.py` — viewer 경량화(PNG→JPG)
- `scripts/build_plotly_viewer.py` — viewer 생성 (DATASETS registry 에 canonical run 경로)
- `scripts/wandb_skill_explorer.py` — wandb 업로드 (참고)
- `results/method_comparison/` — 실험1 산출물 (png, metrics.csv)
- 메모리: `~/.claude/projects/-home-iw-code-lotus-test/memory/` (project_lotus_fork, viewer_hosting, user_profile, reference_dataset_layout)

---

## 9. 사용자 컨텍스트

- 한양대 로보틱스 연구자 (khnuri@hanyang.ac.kr). **한국어 설명 선호.**
- GitHub: `khnuri-cell`(본인), `kimz1121`(동료/현재 git remote).
