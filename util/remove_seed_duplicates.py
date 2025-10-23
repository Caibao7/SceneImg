#!/usr/bin/env python3
"""根据种子图片，剔除下载目录中与种子几乎完全一致的图片。

默认使用 dHash，阈值为 2（可调整），当相似度过高时删除图片及对应 JSON 元数据。
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
from typing import Iterable

from PIL import Image, ImageOps

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def sanitize(text: str) -> str:
    import re

    slug = re.sub(r"[^\w\-\u4e00-\u9fff]+", "-", text.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    if not slug:
        slug = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]
    return slug


def gather_images(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for suffix in IMAGE_SUFFIXES:
        yield from root.rglob(f"*{suffix}")


def compute_dhash(path: pathlib.Path, hash_size: int = 8) -> int:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        diff = []
        for y in range(hash_size):
            for x in range(hash_size):
                diff.append(img.getpixel((x, y)) > img.getpixel((x + 1, y)))
        value = 0
        for i, v in enumerate(diff):
            if v:
                value |= 1 << i
        return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def remove_duplicates_for_seed(
    seed_path: pathlib.Path,
    output_root: pathlib.Path,
    threshold: int,
    dry_run: bool,
) -> int:
    slug = sanitize(seed_path.stem)
    seed_dir = output_root / slug
    if not seed_dir.exists():
        return 0

    try:
        seed_hash = compute_dhash(seed_path)
    except Exception as exc:
        print(f"[warn] 无法读取种子 {seed_path}: {exc}")
        return 0

    removed = 0
    for image_path in gather_images(seed_dir):
        try:
            candidate_hash = compute_dhash(image_path)
        except Exception as exc:
            print(f"[warn] 读取 {image_path} 失败: {exc}")
            continue

        dist = hamming(seed_hash, candidate_hash)
        if dist <= threshold:
            removed += 1
            print(f"[info] 删除 {image_path} (dHash 距离={dist})")
            if not dry_run:
                image_path.unlink(missing_ok=True)
                meta_path = image_path.with_suffix(".json")
                meta_path.unlink(missing_ok=True)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="删除与种子几乎相同的下载图片")
    parser.add_argument(
        "--seed-dir",
        default="crawledimg/search_by_images/seed",
        help="种子图片目录（默认 crawledimg/search_by_images/seed）",
    )
    parser.add_argument(
        "--output-dir",
        default="crawledimg/search_by_images/baidu",
        help="下载图片根目录（默认 crawledimg/search_by_images/baidu）",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="dHash 距离阈值，小于等于该值判定为重复（默认 2）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅输出将删除的文件，不实际删除")
    args = parser.parse_args()

    seed_root = pathlib.Path(args.seed_dir).resolve()
    output_root = pathlib.Path(args.output_dir).resolve()
    if not seed_root.exists():
        raise SystemExit(f"种子目录不存在：{seed_root}")
    if not output_root.exists():
        raise SystemExit(f"下载目录不存在：{output_root}")

    total_removed = 0
    seeds = list(gather_images(seed_root))
    if not seeds:
        print(f"未找到任何种子图片：{seed_root}")
        return

    for seed_path in seeds:
        total_removed += remove_duplicates_for_seed(seed_path, output_root, args.threshold, args.dry_run)

    action = "将删除" if args.dry_run else "已删除"
    print(f"{action} {total_removed} 张重复图片。")


if __name__ == "__main__":
    main()
