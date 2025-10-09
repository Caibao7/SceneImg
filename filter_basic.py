#!/usr/bin/env python3
"""使用 CLIP 对爬取的图片进行初步筛选。"""

import argparse
import csv
import pathlib
from typing import Iterable, Optional

import clip
import torch
from PIL import Image
from torchvision.transforms import Compose
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights

PROMPTS = {
    "indoor": "a photo of an indoor living space with natural lighting",
    "outdoor": "a photo of an outdoor landscape",
    "tidy": "a perfectly tidy minimalist interior with few objects",
    "messy": "a very messy cluttered interior with many objects scattered everywhere",
    "moderate": "a realistic indoor scene with a manageable amount of objects and some clutter",
}


def load_model(device: str):
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(list(PROMPTS.values())).to(device)
    with torch.no_grad():
        text_embed = model.encode_text(text_tokens)
        text_embed /= text_embed.norm(dim=-1, keepdim=True)
    return model, preprocess, text_embed


def compute_scores(
    model,
    preprocess,
    text_embed,
    device: str,
    image_path: pathlib.Path,
) -> dict:
    pil = Image.open(image_path).convert("RGB")
    try:
        image = preprocess(pil).unsqueeze(0).to(device)
    finally:
        pil.close()
    with torch.no_grad():
        img_embed = model.encode_image(image)
        img_embed /= img_embed.norm(dim=-1, keepdim=True)
        sims = (img_embed @ text_embed.T).cpu().numpy()[0]
    keys = list(PROMPTS.keys())
    return {key: float(sims[idx]) for idx, key in enumerate(keys)}


def should_keep(scores: dict, indoor_margin: float, moderate_margin: float) -> bool:
    if scores["indoor"] < scores["outdoor"] + indoor_margin:
        return False
    # 临时关闭 tidy/messy/moderate 相关的过滤逻辑
    # if scores["moderate"] < max(scores["tidy"], scores["messy"]) + moderate_margin:
    #     return False
    return True


def count_objects(
    detector,
    transform: Compose,
    image_path: pathlib.Path,
    device: str,
    score_threshold: float,
) -> int:
    pil = Image.open(image_path).convert("RGB")
    try:
        tensor = transform(pil).to(device)
    finally:
        pil.close()
    with torch.no_grad():
        outputs = detector([tensor])[0]
    scores = outputs.get("scores")
    if scores is None:
        return 0
    return int((scores >= score_threshold).sum())


def gather_images(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        yield from root.rglob(pattern)


def filter_images(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    device: str,
    indoor_margin: float,
    moderate_margin: float,
    limit: int | None,
    object_count_min: Optional[int],
    object_count_max: Optional[int],
    detector_threshold: float,
) -> None:
    model, preprocess, text_embed = load_model(device)
    keys = list(PROMPTS.keys())
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "clip_filter_log.csv"

    need_object_filter = object_count_min is not None or object_count_max is not None
    detector = None
    detector_transform: Optional[Compose] = None
    if need_object_filter:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        detector = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
        detector_transform = weights.transforms()

    kept = 0
    with log_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = ["path", *keys]
        if need_object_filter:
            header.append("object_count")
        writer.writerow(header)
        for image_path in gather_images(input_dir):
            scores = compute_scores(model, preprocess, text_embed, device, image_path)
            if not should_keep(scores, indoor_margin, moderate_margin):
                continue
            object_count = None
            if need_object_filter and detector and detector_transform:
                object_count = count_objects(detector, detector_transform, image_path, device, detector_threshold)
                if object_count_min is not None and object_count < object_count_min:
                    continue
                if object_count_max is not None and object_count > object_count_max:
                    continue
            target = output_dir / image_path.name
            target.write_bytes(image_path.read_bytes())
            row = [str(image_path), *(scores[key] for key in keys)]
            if need_object_filter:
                row.append(object_count)
            writer.writerow(row)
            kept += 1
            if limit and kept >= limit:
                break
    print(f"筛选完成，共保留 {kept} 张图片")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP 初筛脚本")
    parser.add_argument("-i", "--input", default="crawledimg/baidu/raw", help="待筛选图片目录")
    parser.add_argument("-o", "--output", default="filteredimg/baidu/auto", help="输出目录")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--indoor-margin", type=float, default=0.02, help="室内得分需要领先室外的 margin")
    parser.add_argument(
        "--moderate-margin",
        type=float,
        default=0,
        help="适度杂乱得分需要领先 tidy/messy 的 margin",
    )
    parser.add_argument("--limit", type=int, help="最多保留图片数量，可选")
    parser.add_argument(
        "-l", 
        "--object-count-min",
        type=int,
        help="检测到的物体数量下限，未提供则不限制",
    )
    parser.add_argument(
        "-u", 
        "--object-count-max",
        type=int,
        help="检测到的物体数量上限，未提供则不限制",
    )
    parser.add_argument(
        "--detector-threshold",
        type=float,
        default=0.5,
        help="物体检测置信度阈值（默认 0.5）",
    )
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    if (
        args.object_count_min is not None
        and args.object_count_max is not None
        and args.object_count_min > args.object_count_max
    ):
        raise SystemExit("object-count-min 不能大于 object-count-max。")
    filter_images(
        input_dir,
        output_dir,
        args.device,
        args.indoor_margin,
        args.moderate_margin,
        args.limit,
        args.object_count_min,
        args.object_count_max,
        args.detector_threshold,
    )


if __name__ == "__main__":
    main()
