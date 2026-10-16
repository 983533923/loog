#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给图片叠加百分比网格，便于读出落位坐标。

    python grid_overlay.py <image> <out.png> [--max-size 1200] [--step 5]

输出的网格线按整幅图的百分比分布（0/5/10/.../100），边缘标有百分比数字。
把网格图交给视觉判断，就能直接给出 center / shirt_w 这类归一化数值。
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main(argv=None):
    ap = argparse.ArgumentParser(description="叠加百分比网格的预览图")
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--max-size", type=int, default=1200)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args(argv)

    img = Image.open(args.image).convert("RGB")
    scale = min(1.0, args.max_size / max(img.size))
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.LANCZOS)
    canvas = img.copy()
    dr = ImageDraw.Draw(canvas)
    w, h = canvas.size

    for pct in range(0, 101, args.step):
        x = min(w - 1, round(pct / 100 * w))
        y = min(h - 1, round(pct / 100 * h))
        major = pct % 10 == 0
        color = (255, 60, 60) if major else (255, 190, 60)
        dr.line([(x, 0), (x, h)], fill=color, width=2 if major else 1)
        dr.line([(0, y), (w, y)], fill=color, width=2 if major else 1)
        if major:
            dr.text((x + 3, 3), f"{pct}", fill=(255, 255, 0))
            dr.text((3, y + 3), f"{pct}", fill=(255, 255, 0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"{args.image} {Image.open(args.image).size} -> {args.out} {canvas.size}")


if __name__ == "__main__":
    main()
