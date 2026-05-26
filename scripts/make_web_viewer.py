#!/usr/bin/env python3
"""Produce a size-optimized copy of viewer/ for GitHub Pages.

Converts the PNG frame thumbnails / grids to JPEG (quality 82) which keeps
visual quality but cuts the dominant `frames_big` payload to ~1/5, while
leaving the already-small H.264 mp4 videos untouched. The two JS references
(`frames/${i}.png`, `frames_big/${i}.png`) are rewritten to `.jpg`.

Source viewer/ is left fully intact; output goes to viewer_web/.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from PIL import Image

SRC = Path("viewer")
DST = Path("viewer_web")
QUALITY = 82
DATASETS = ["libero_object", "libero_goal", "robocasa365_atomic_seen"]


def convert_dir(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in sorted(src_dir.glob("*.png")):
        img = Image.open(png).convert("RGB")
        img.save(dst_dir / (png.stem + ".jpg"), "JPEG", quality=QUALITY, optimize=True)
        n += 1
    return n


def main() -> None:
    if not SRC.exists():
        sys.exit(f"missing {SRC}/ (run from repo root)")
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir()

    for ds in DATASETS:
        s = SRC / ds
        d = DST / ds
        if not s.exists():
            print(f"  skip {ds} (not built)")
            continue
        print(f"=== {ds} ===")
        t0 = time.time()
        nf = convert_dir(s / "frames", d / "frames")
        nb = convert_dir(s / "frames_big", d / "frames_big")
        # videos copied verbatim (already small H.264)
        shutil.copytree(s / "videos", d / "videos")
        # html with .png -> .jpg on the two image references
        html = (s / "index.html").read_text()
        html = html.replace("`frames/${i}.png`", "`frames/${i}.jpg`")
        html = html.replace('"frames_big/${i}.png"', '"frames_big/${i}.jpg"')
        (d / "index.html").write_text(html)
        print(f"  frames={nf} frames_big={nb} videos copied  ({time.time()-t0:.0f}s)")

    # root index
    root = SRC / "index.html"
    if root.exists():
        shutil.copy(root, DST / "index.html")
    print("done.")


if __name__ == "__main__":
    main()
