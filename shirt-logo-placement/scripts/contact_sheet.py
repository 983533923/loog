# -*- coding: utf-8 -*-
"""把一批图片拼成一张总览图，便于快速核对批量处理结果。

    python contact_sheet.py <目录或图片...> --out sheet.png [--cell 360] [--cols 4]

每格是一张缩略图，下方标注相对文件名。用于一次性检查「哪些贴上了、位置对不对」。
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect(targets):
    files = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in SUPPORTED))
        elif p.suffix.lower() in SUPPORTED:
            files.append(p)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成多图总览")
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=360)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args(argv)

    files = collect(args.targets)
    if not files:
        print("没有找到图片")
        return 1

    cell, label_h, pad = args.cell, 20, 8
    cols = max(1, args.cols)
    rows = (len(files) + cols - 1) // cols
    W = cols * cell + pad * (cols + 1)
    H = rows * (cell + label_h) + pad * (rows + 1)
    sheet = Image.new("RGB", (W, H), (24, 24, 28))
    dr = ImageDraw.Draw(sheet)

    for i, f in enumerate(files):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + label_h + pad)
        try:
            img = Image.open(f).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            dr.text((x + 4, y + 4), f"打开失败: {exc}", fill=(255, 120, 120))
            continue
        img.thumbnail((cell, cell), Image.LANCZOS)
        sheet.paste(img, (x + (cell - img.width) // 2, y + (cell - img.height) // 2))
        dr.text((x + 4, y + cell + 4), f.name[:40], fill=(225, 225, 225))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"{len(files)} 张 -> {args.out} {sheet.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
