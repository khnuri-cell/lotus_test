"""Convert the 18 RoboCasa365 v1.0 target/atomic-seen LeRobot **v2.1** datasets
into LIBERO/robomimic-style HDF5s that the LOTUS skill-discovery pipeline reads.

This is a *minimal* v2.1 reader that bypasses the lerobot library entirely
(lerobot 0.4+ refuses v2.1; lerobot 0.3.x requires torchcodec which needs
FFmpeg shared libs unavailable in the container). We read:
    meta/info.json                       - schema
    meta/episodes.jsonl                  - episode metadata (length, tasks[])
    meta/tasks.jsonl                     - task_index -> natural-language str
    data/chunk-XXX/episode_NNNNNN.parquet - per-episode action/state
    videos/chunk-XXX/<cam>/episode_NNNNNN.mp4 - per-episode camera video

State layout assumed (PandaOmron 16-D, confirmed via inspection):
    [0..6]   robot0_base                  → dropped
    [7..9]   robot0_eef_pos               → obs/ee_states[0:3]
    [10..13] robot0_eef_quat (xyzw)       → obs/ee_states[3:6]   (xyz of quat; w dropped)
    [14..15] robot0_gripper_qpos          → obs/gripper_states
    joint_states                          → zeros(7)            (not provided)

Usage:
    python scripts/convert_robocasa365_atomic_seen.py \\
        --src-root /robocasa/datasets/v1.0/target/atomic \\
        --out-dir  datasets/robocasa365_atomic_seen \\
        --image-size 128 \\
        --max-demos-per-task 25
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd


ATOMIC_SEEN_18 = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]

EEF_POS_SLICE   = slice(7, 10)    # PandaOmron 16-D state
EEF_QUAT_SLICE  = slice(10, 13)   # xyz of xyzw (drop w)
GRIPPER_SLICE   = slice(14, 16)


def latest_lerobot_dir(task_dir: Path) -> Path:
    """Each task dir has YYYYMMDD subdirs; pick the most recent with lerobot/."""
    cands = sorted(
        (p for p in task_dir.iterdir() if p.is_dir() and (p / "lerobot" / "meta" / "info.json").exists()),
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"no dated lerobot subdir under {task_dir}")
    return cands[0] / "lerobot"


def read_episodes_jsonl(path: Path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_tasks_jsonl(path: Path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[int(d["task_index"])] = d["task"]
    return out


def decode_mp4(path: Path, image_size: int | None, expected_T: int) -> np.ndarray:
    """Decode an mp4 → (T, H, W, 3) uint8 array (RGB). Optionally resize square."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # cv2 returns BGR; lotus expects RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if image_size and (frame.shape[0] != image_size or frame.shape[1] != image_size):
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    arr = np.stack(frames, axis=0)
    # Length sanity vs parquet; some encoders drop +/-1 frame. Trim or pad to expected.
    if arr.shape[0] != expected_T:
        if arr.shape[0] > expected_T:
            arr = arr[:expected_T]
        else:
            pad = expected_T - arr.shape[0]
            arr = np.concatenate([arr, np.repeat(arr[-1:], pad, axis=0)], axis=0)
    return arr


def collect_episode_v21(lerobot_root: Path, ep_idx: int, ep_meta: dict, tasks_map: dict,
                        image_size: int | None):
    chunk = ep_idx // 1000
    chunk_str = f"chunk-{chunk:03d}"
    ep_str = f"episode_{ep_idx:06d}"

    parquet_path = lerobot_root / "data" / chunk_str / f"{ep_str}.parquet"
    df = pd.read_parquet(parquet_path)
    T = len(df)

    state = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 16)
    action = np.stack(df["action"].values).astype(np.float32)             # (T, 12)

    ee_pos = state[:, EEF_POS_SLICE]                  # (T, 3)
    ee_quat_xyz = state[:, EEF_QUAT_SLICE]            # (T, 3) — drop w
    ee_states = np.concatenate([ee_pos, ee_quat_xyz], axis=1).astype(np.float32)
    gripper = state[:, GRIPPER_SLICE].astype(np.float32)
    joint_states = np.zeros((T, 7), dtype=np.float32)

    # Cameras
    cam_root = lerobot_root / "videos" / chunk_str
    agentview = decode_mp4(cam_root / "observation.images.robot0_agentview_left" / f"{ep_str}.mp4",
                            image_size, T)
    eye = decode_mp4(cam_root / "observation.images.robot0_eye_in_hand" / f"{ep_str}.mp4",
                     image_size, T)

    # Per-demo task instruction (prefer episodes.jsonl tasks[0] if present)
    instr = None
    if "tasks" in ep_meta and ep_meta["tasks"]:
        instr = ep_meta["tasks"][0]
    if not instr and "task_index" in df.columns:
        ti = int(df["task_index"].iloc[0])
        instr = tasks_map.get(ti)

    return {
        "agentview_rgb": agentview,
        "eye_in_hand_rgb": eye,
        "ee_states": ee_states,
        "gripper_states": gripper,
        "joint_states": joint_states,
        "state": state,
        "action": action,
        "instruction": instr or "",
    }


def write_demo(h5_data_grp: h5py.Group, demo_idx: int, ep: dict):
    T = ep["action"].shape[0]
    demo = h5_data_grp.create_group(f"demo_{demo_idx}")
    demo.attrs["num_samples"] = T
    demo.attrs["model_file"] = ""
    demo.attrs["task_instruction"] = ep["instruction"]

    demo.create_dataset("actions", data=ep["action"], compression="gzip", compression_opts=4)
    demo.create_dataset("dones", data=np.zeros((T,), dtype=np.int64))
    demo.create_dataset("rewards", data=np.zeros((T,), dtype=np.float32))
    demo.create_dataset("states", data=ep["state"], compression="gzip", compression_opts=4)
    demo.create_dataset("robot_states", data=ep["state"], compression="gzip", compression_opts=4)

    obs = demo.create_group("obs")
    obs.create_dataset("agentview_rgb", data=ep["agentview_rgb"],
                       compression="gzip", compression_opts=4,
                       chunks=(1, *ep["agentview_rgb"].shape[1:]))
    obs.create_dataset("eye_in_hand_rgb", data=ep["eye_in_hand_rgb"],
                       compression="gzip", compression_opts=4,
                       chunks=(1, *ep["eye_in_hand_rgb"].shape[1:]))
    obs.create_dataset("joint_states", data=ep["joint_states"])
    obs.create_dataset("gripper_states", data=ep["gripper_states"])
    obs.create_dataset("ee_states", data=ep["ee_states"])


def convert_one_task(task: str, lerobot_root: Path, out_path: Path,
                     image_size: int | None, max_demos: int | None):
    print(f"\n[{task}] lerobot_root={lerobot_root}")
    info = json.load(open(lerobot_root / "meta" / "info.json"))
    eps = read_episodes_jsonl(lerobot_root / "meta" / "episodes.jsonl")
    tasks_map = read_tasks_jsonl(lerobot_root / "meta" / "tasks.jsonl")
    eps.sort(key=lambda e: int(e["episode_index"]))

    n_ep_total = info.get("total_episodes", len(eps))
    n_ep = n_ep_total if (max_demos is None or max_demos <= 0) else min(n_ep_total, max_demos)
    print(f"[{task}]   episodes total={n_ep_total} using={n_ep}  fps={info.get('fps')}")
    print(f"[{task}]   image_size={image_size or 'native'}  tasks_map_size={len(tasks_map)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_frames = 0
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("data")
        grp.attrs["env_args"] = json.dumps({"env_name": task, "type": 1, "env_kwargs": {}})
        grp.attrs["problem_info"] = json.dumps(
            {"language_instruction": task, "problem_name": task}
        )
        grp.attrs["total"] = 0

        for ep_idx in range(n_ep):
            ep = collect_episode_v21(lerobot_root, ep_idx, eps[ep_idx], tasks_map, image_size)
            write_demo(grp, ep_idx, ep)
            total_frames += ep["action"].shape[0]
            if ep_idx % 10 == 0 or ep_idx == n_ep - 1:
                dt = time.time() - t0
                rate = (ep_idx + 1) / max(dt, 1e-6)
                eta = (n_ep - ep_idx - 1) / max(rate, 1e-6)
                print(f"[{task}]   {ep_idx+1}/{n_ep}  cum_frames={total_frames}  "
                      f"{rate:.1f} ep/s  eta={eta:.0f}s")

        grp.attrs["total"] = total_frames

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[{task}] DONE → {out_path}  ({n_ep} demos, {total_frames} frames, "
          f"{size_mb:.1f} MB, {time.time()-t0:.1f}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=Path,
                   default=Path("/robocasa/datasets/v1.0/target/atomic"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("datasets/robocasa365_atomic_seen"))
    p.add_argument("--image-size", type=int, default=128,
                   help="Resize images to NxN (LIBERO=128). 0 = keep native 256.")
    p.add_argument("--max-demos-per-task", type=int, default=0,
                   help="0 = all (~500 per task).")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Subset of task names. Default = all 18.")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    image_size = args.image_size if args.image_size > 0 else None
    max_demos = args.max_demos_per_task if args.max_demos_per_task > 0 else None
    tasks = args.tasks or ATOMIC_SEEN_18
    unknown = [t for t in tasks if t not in ATOMIC_SEEN_18]
    if unknown:
        raise SystemExit(f"unknown atomic-seen task(s): {unknown}")

    print(f"[CONVERT] {len(tasks)} task(s) → {args.out_dir.resolve()}")
    print(f"  image_size={image_size or 'native'}  max_demos={max_demos or 'all'}")

    t0 = time.time()
    for i, task in enumerate(tasks):
        out_path = args.out_dir / f"{task}.hdf5"
        if args.skip_existing and out_path.exists():
            print(f"\n[{task}] SKIP (exists)")
            continue
        task_src = args.src_root / task
        if not task_src.exists():
            print(f"\n[{task}] WARN: missing — {task_src}")
            continue
        lerobot_root = latest_lerobot_dir(task_src)
        convert_one_task(task, lerobot_root, out_path, image_size, max_demos)
        print(f"  ===> {i+1}/{len(tasks)} done, total elapsed {time.time()-t0:.1f}s")

    print(f"\nALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
