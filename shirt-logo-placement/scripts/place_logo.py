#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""服装上衣 logo 贴合引擎（确定性局部合成，不重绘衣服）。

按 plan 指定的落位把 logo 贴到服装图上，默认**就地覆盖原图**。只在 logo 外扩区域内
做像素运算，因此即便原图很大也不会占用大量内存。

推荐用法（项目级批量、就地替换）：
    python place_logo.py --logo logo.png --root <项目根目录> --plan plan.json --inplace

其它用法：
    # 单图
    python place_logo.py --logo logo.png --image a.jpg --out a_logo.jpg \
        --center 0.60,0.29 --width 0.09 --angle 0

    # 只输出到别的目录、保持相同相对结构（预览用，不覆盖原图）
    python place_logo.py --logo logo.png --root <项目根目录> --plan plan.json --out out/

落位描述（plan 里每个条目的字段）：
    center   [cx, cy]      归一化(<=1.5)或像素
    width    W             logo 宽度：<=1.5 视为占图宽比例，否则像素
    或            shirt_w 衣服宽度（归一化或像素）+ ratio，logo 宽度 = shirt_w * ratio
                            —— 同款图片统一用同一个 ratio，可保证各图 logo 视觉大小一致
    angle    deg           顺时针为正，0 = 水平
    quad     [[x,y]*4]     直接给四角（左上→右上→右下→左下），优先级最高
    skip     true          跳过该图，可附 reason

一张图里有**多件衣服**时，用 placements 列出多个落位（数组形式亦可）：
    {"center": ..., "shirt_w": ...}                        # 单件
    {"placements": [{"center": ...}, {"center": ...}]}     # 多件，逐个贴合

plan.json 结构（键为图片相对 --root 的路径，亦可用纯文件名）：
    {
      "defaults": {"ratio": 0.088},
      "款式A/主图/01.jpg": {"center": [0.60, 0.29], "shirt_w": 0.72},
      "款式A/详情图/02.jpg": {"center": [0.62, 0.30], "shirt_w": 0.68, "angle": -8},
      "款式A/详情图/03.jpg": {"skip": true, "reason": "背面展示图"}
    }
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------ logo 素材

def load_logo(path: Path) -> Image.Image:
    """读取 logo 并按内容裁边；没有透明通道时用背景色差异生成 alpha。"""
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img)
    if arr[..., 3].min() >= 250:  # 例如白底图：用与背景色的差异造 alpha
        rgb = arr[..., :3].astype(np.int16)
        corner = np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3),
                                 rgb[-8:, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)])
        bg = np.median(corner, axis=0)
        dist = np.abs(rgb - bg).max(axis=2).astype(np.float32)
        arr = arr.copy()
        arr[..., 3] = np.clip(dist * (255.0 / max(1.0, float(dist.max()))) * 1.25, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGBA")
    arr = np.asarray(img)
    ys, xs = np.where(arr[..., 3] > 8)
    if len(xs):
        img = img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return img


# ------------------------------------------------------------------ 几何

def quad_from_center(cx, cy, width, aspect, angle_deg):
    """按中心、宽度、宽高比、顺时针角度生成四角（左上→右上→右下→左下）。"""
    half_w, half_h = width / 2.0, width / aspect / 2.0
    t = math.radians(angle_deg)
    cos_t, sin_t = math.cos(t), math.sin(t)
    pts = []
    for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
        pts.append((cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t))
    return pts


def perspective_coeffs(dst_quad, src_quad):
    a, b = [], []
    for (xd, yd), (xs, ys) in zip(dst_quad, src_quad):
        a.append([xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd])
        b.append(xs)
        a.append([0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd])
        b.append(ys)
    return tuple(np.linalg.solve(np.asarray(a, np.float64), np.asarray(b, np.float64)))


def _px(value, extent):
    """归一化(<=1.5)转像素；否则按像素处理。"""
    return float(value) * extent if abs(float(value)) <= 1.5 else float(value)


def resolve_quad(spec: dict, size, aspect: float):
    """把落位描述解析为像素四点。"""
    w, h = size
    if spec.get("quad"):
        quad = [(float(p[0]), float(p[1])) for p in spec["quad"]]
        if max(max(abs(p[0]), abs(p[1])) for p in quad) <= 1.5:
            quad = [(p[0] * w, p[1] * h) for p in quad]
        return quad

    cx = _px(spec["center"][0], w)
    cy = _px(spec["center"][1], h)
    if "width" in spec:
        width = _px(spec["width"], w)
    elif "shirt_w" in spec:
        ratio = float(spec.get("ratio", 0.088))
        width = _px(spec["shirt_w"], w) * ratio
    else:
        raise ValueError("落位需要 quad / width / (shirt_w + ratio) 之一")
    return quad_from_center(cx, cy, width, aspect, float(spec.get("angle", 0.0)))


# ------------------------------------------------------------------ 合成

def _blur(arr, radius):
    if radius <= 0:
        return arr.astype(np.float32)
    return np.asarray(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)


def _luma(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def blend(base, logo, mode):
    if mode == "multiply":
        return base * logo / 255.0
    if mode == "screen":
        return 255.0 - (255.0 - base) * (255.0 - logo) / 255.0
    if mode == "overlay":
        return np.where(base <= 127.5, 2.0 * base * logo / 255.0,
                        255.0 - 2.0 * (255.0 - base) * (255.0 - logo) / 255.0)
    return logo


def composite(base_img: Image.Image, logo: Image.Image, quad, *, mode="auto",
              opacity=1.0, light_follow=0.5, texture_keep=0.10, feather=None,
              shadow=0.0):
    """把 logo 贴合到 base_img 上，只在 logo 外扩区域内运算。返回 (结果图, 诊断信息)。"""
    base = base_img.convert("RGB")
    W, H = base.size
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    pad = max(2.0, span * 0.05 + 2.0)
    x0 = int(max(0, math.floor(min(xs) - pad)))
    y0 = int(max(0, math.floor(min(ys) - pad)))
    x1 = int(min(W, math.ceil(max(xs) + pad)))
    y1 = int(min(H, math.ceil(max(ys) + pad)))
    roi_size = (x1 - x0, y1 - y0)
    roi = base.crop((x0, y0, x1, y1))

    quad_roi = [(px - x0, py - y0) for px, py in quad]
    src_quad = [(0.0, 0.0), (float(logo.width), 0.0),
                (float(logo.width), float(logo.height)), (0.0, float(logo.height))]
    warped = logo.transform(roi_size, Image.PERSPECTIVE,
                            perspective_coeffs(quad_roi, src_quad),
                            resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))

    base_arr = np.asarray(roi, dtype=np.float32)
    logo_arr = np.asarray(warped.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(warped.getchannel("A"), dtype=np.float32) / 255.0
    weight = float(max(alpha.sum(), 1e-6))
    base_luma = float((_luma(base_arr) * alpha).sum() / weight)
    logo_luma = float((_luma(logo_arr) * alpha).sum() / weight)

    resolved = mode
    if mode == "auto":
        if base_luma >= 168:
            resolved = "multiply" if logo_luma < 168 else "normal"
        elif base_luma <= 92:
            resolved = "screen" if logo_luma > 92 else "normal"
        else:
            resolved = "normal"

    low = _blur(base_arr, max(1.5, span / 14.0))
    mean_low = float(np.clip((_luma(low) * alpha).sum() / weight, 1.0, 255.0))
    illum = 1.0 + (np.clip(_luma(low) / mean_low, 0.55, 1.75) - 1.0) * float(light_follow)
    detail = (base_arr - low).mean(axis=2, keepdims=True) * float(texture_keep)

    merged_region = blend(base_arr, np.clip(logo_arr * illum[..., None] + detail, 0, 255), resolved)

    a = np.clip(alpha * float(opacity), 0.0, 1.0)
    if shadow > 0:
        off = max(1, int(round(span * 0.012)))
        sh = np.zeros_like(a)
        sh[off:, off:] = a[:-off, :-off]
        merged_region = merged_region * (1.0 - float(shadow) * sh[..., None])
    r = float(feather) if feather is not None else max(0.5, span / 160.0)
    a = np.clip(_blur(a * 255.0, r) / 255.0 * 1.06, 0, 1)

    merged = base_arr * (1.0 - a[..., None]) + merged_region * a[..., None]
    out = base.copy()
    out.paste(Image.fromarray(np.clip(merged, 0, 255).astype(np.uint8), "RGB"), (x0, y0))

    info = {
        "logo_box_px": [int(round(min(xs))), int(round(min(ys))),
                        int(round(max(xs))), int(round(max(ys)))],
        "logo_size_px": [round(max(xs) - min(xs), 1), round(max(ys) - min(ys), 1)],
        "logo_w_ratio_of_image": round((max(xs) - min(xs)) / W, 4),
        "center_norm": [round((min(xs) + max(xs)) / 2 / W, 4),
                        round((min(ys) + max(ys)) / 2 / H, 4)],
        "mode": resolved,
        "feather_px": round(r, 2),
    }
    return out, info


# ------------------------------------------------------------------ 驱动

def save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(path, quality=95, subsampling=0)
    else:
        img.save(path)


def process_file(src: Path, logo: Image.Image, spec: dict, dst: Path):
    specs = normalize_specs(spec)
    with Image.open(src) as img:
        result = img.convert("RGB")
        infos = []
        for one in specs:
            quad = resolve_quad(one, result.size, logo.width / logo.height)
            params = {k: one[k] for k in ("mode", "opacity", "light_follow", "texture_keep",
                                          "feather", "shadow") if k in one}
            result, info = composite(result, logo, quad, **params)
            infos.append(info)
        save_image(result, dst)
    if len(infos) == 1:
        return infos[0]
    return {"placements": infos, "count": len(infos)}


def normalize_specs(spec):
    """把条目规范化成落位列表，支持单件、placements、数组三种写法。"""
    if isinstance(spec, list):
        return spec
    if "placements" in spec:
        base = {k: v for k, v in spec.items() if k != "placements"}
        return [{**base, **p} for p in spec["placements"]]
    return [spec]


def run_batch(root: Path, logo: Image.Image, plan: dict, out_dir: Path | None, inplace: bool):
    defaults = dict(plan.get("defaults", {}))
    entries = {k: v for k, v in plan.items() if k != "defaults"}
    reports = []
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in SUPPORTED
                   and not p.stem.endswith("_logo"))
    for src in files:
        rel = src.relative_to(root).as_posix()
        spec = entries.get(rel) or entries.get(src.name)
        if spec is None:
            reports.append({"file": rel, "status": "no_plan"})
            continue
        merged = {**defaults, **spec}
        if merged.get("skip"):
            reports.append({"file": rel, "status": "skipped", "reason": merged.get("reason", "")})
            continue
        dst = src if inplace else (out_dir / rel)
        try:
            info = process_file(src, logo, merged, dst)
        except Exception as exc:  # noqa: BLE001
            reports.append({"file": rel, "status": "error", "error": str(exc)})
            continue
        reports.append({"file": rel, "status": "ok", "out": str(dst), **info})
    return reports


def main(argv=None):
    ap = argparse.ArgumentParser(description="服装上衣 logo 贴合引擎")
    ap.add_argument("--logo", required=True, help="logo 素材（透明底 PNG 最佳）")
    ap.add_argument("--root", help="项目根目录：递归处理其中的图片")
    ap.add_argument("--image", help="单张图片路径")
    ap.add_argument("--plan", help="plan.json 落位表（批量模式推荐）")
    ap.add_argument("--out", help="输出路径：批量模式为目录；单图模式为文件")
    ap.add_argument("--inplace", action="store_true", help="就地覆盖原图（推荐）")
    ap.add_argument("--report", help="把 JSON 报告写到该文件")
    ap.add_argument("--quad")
    ap.add_argument("--center")
    ap.add_argument("--width", type=float)
    ap.add_argument("--shirt-w", type=float)
    ap.add_argument("--ratio", type=float, default=0.088)
    ap.add_argument("--angle", type=float, default=0.0)
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "normal", "multiply", "screen", "overlay"])
    ap.add_argument("--light-follow", type=float, default=0.5)
    ap.add_argument("--texture-keep", type=float, default=0.10)
    ap.add_argument("--feather", type=float, default=None)
    ap.add_argument("--shadow", type=float, default=0.0)
    args = ap.parse_args(argv)

    logo = load_logo(Path(args.logo))
    common = {"mode": args.mode, "light_follow": args.light_follow,
              "texture_keep": args.texture_keep, "shadow": args.shadow}
    if args.feather is not None:
        common["feather"] = args.feather

    if args.root:
        root = Path(args.root)
        if not root.is_dir():
            print(f"目录不存在：{root}", file=sys.stderr)
            return 2
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else \
            {"defaults": common}
        if args.plan:
            plan.setdefault("defaults", {}).update(common)
        if not args.inplace and not args.out:
            print("批量模式需要 --inplace 或 --out", file=sys.stderr)
            return 2
        reports = run_batch(root, logo, plan, Path(args.out) if args.out else None, args.inplace)
        summary = {"logo": str(args.logo), "root": str(root),
                   "total": len(reports),
                   "ok": sum(r["status"] == "ok" for r in reports),
                   "skipped": sum(r["status"] == "skipped" for r in reports),
                   "no_plan": sum(r["status"] == "no_plan" for r in reports),
                   "error": sum(r["status"] == "error" for r in reports)}
        payload = {"summary": summary, "reports": reports}
        if args.report:
            Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0

    if not args.image:
        print("需要 --image 或 --root", file=sys.stderr)
        return 2
    spec = dict(common)
    if args.quad:
        spec["quad"] = [[float(v) for v in chunk.split(",")]
                        for chunk in args.quad.replace(";", " ").split()]
    elif args.center and args.width:
        spec["center"] = [float(v) for v in args.center.split(",")]
        spec["width"] = args.width
    elif args.center and args.shirt_w:
        spec["center"] = [float(v) for v in args.center.split(",")]
        spec["shirt_w"] = args.shirt_w
        spec["ratio"] = args.ratio
    else:
        print("需要 --quad，或 --center 搭配 --width / --shirt-w", file=sys.stderr)
        return 2
    spec["angle"] = args.angle

    src = Path(args.image)
    dst = src if args.inplace else Path(args.out or f"{src.stem}_logo{src.suffix}")
    info = process_file(src, logo, spec, dst)
    payload = {"file": str(src), "status": "ok", "out": str(dst), **info}
    if args.report:
        Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
