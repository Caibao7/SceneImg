#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用多模态 VLM 对候选室内场景图片做筛选。

设计目标：
1. 输入形式灵活：支持 JSON/JSONL 清单或直接扫描目录；
2. 输出标准化：写出 JSONL，每行包含筛选结果、评分与原因；
3. 模型接口：兼容 OpenAI Responses API、Ollama 本地与云端模型；
4. Prompt 针对“适度杂乱、含丰富可操作物体的室内场景”，以支撑机器人任务生成。
"""

# 以下是一个使用 ollama 库调用 Ollama API 的示例代码片段
# import os
# from ollama import Client

# client = Client(
#     host="https://ollama.com",
#     headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
# )

# messages = [
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ]

# for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
#   print(part['message']['content'], end='', flush=True)

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

try:  # 可选依赖：仅在使用 OpenAI 后端时需要
    from openai import OpenAI  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional
    OpenAI = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 可选依赖
    tqdm = None

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("OPENAI_VLM_MODEL", "gpt-4.1")
DEFAULT_OLLAMA_LOCAL_URL = os.environ.get("OLLAMA_LOCAL_URL", "http://localhost:11434")
DEFAULT_OLLAMA_CLOUD_URL = os.environ.get("OLLAMA_CLOUD_URL", "https://ollama.com")
OLLAMA_API_KEY_ENV = "OLLAMA_API_KEY"
MAX_WORKERS = int(os.environ.get("WORKERS", "2"))
MAX_RETRIES = 3
RETRY_BACKOFF = (1.5, 3.2)  # (base, jitter)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

SYSTEM_PROMPT = r"""
You are an indoor robotics dataset curator. Your task is to decide if an image
shows a **usable, moderately cluttered indoor scene** suitable for robot
manipulation task generation.

### OUTPUT FORMAT (JSON only)
{
  "decision": "keep" | "reject",
  "score": float (0.0–1.0),
  "reason": "≤160 chars concise rationale",
  "room_tags": [string, ...],     # e.g. "office","living room","workbench"
  "object_tags": [string, ...]    # manipulable object types visible, 3–10 items
}

### EVALUATION POLICY
1. **Indoor requirement**
   - Must clearly show an indoor structure: walls, furniture, or fixtures.
   - Reject: outdoor scenes, renders, collages, diagrams.

2. **Viewpoint and framing**
   - Prefer **close or medium views** focusing on a manipulatable working area (for example: tabletop, floor, sofa, chair, closet and etc.).
   - Reject far or wide shots where manipulable areas occupy <30% of the frame.

3. **Object richness**
   - Keep if ≥8 distinct, **reachable** manipulable objects on tables, desks, or floor.
   - Reject if most items are stacked, sealed in boxes, or unreachable.

4. **Clutter balance**
   - Keep: “lived-in”, somewhat messy but **navigable and safe**.
   - Reject if:
     - Overly cluttered (piles, trash, dense disorder, chaos),
     - Too minimal / nearly empty,

5. **Task potential**
   - Favor scenes offering clear robot tasks (pick/place/sort/organize/clean).
   - Reject scenes with inaccessible or ambiguous surfaces.

6. **Image quality**
   - Must be sharp, bright enough to recognize objects.
   - Reject blur, low-light, strong filters, text overlays, or excessive distance.

### SCORING RUBRIC
- 0.8-1.0 strong keep: close indoor shot, 10+ manipulable objects, varied types, good accessibility.
- 0.6-0.79 keep: meets key conditions, clutter or object diversity slightly limited.
- 0.4-0.59 reject: borderline; either too messy, too far, or too few objects.
- <0.4 reject: fails indoor, view, or quality conditions.

### TAGGING GUIDE
- room_tags: coarse scene labels ("office","workbench","living room","kitchen","studio").
- object_tags: common manipulable items ("cup","bottle","book","remote","pen","keyboard","tissue","toy","tool","plate","bowl","phone","charger","snack bag").

Return **JSON only**, compact and valid.
"""

USER_PROMPT_TEMPLATE = r"""
Judge whether this image fits a **usable indoor robotics scene**.

Checklist:
1. Is it clearly an indoor room with visible furniture/walls?
2. Is the main area shown in **close or medium range**, not distant background?
3. Are there ≥8 reachable manipulable objects on tables, shelves, or floor?
4. Is the scene **moderately messy but safe and navigable**, not overfilled or piled?
5. Would it allow diverse robot manipulation tasks (pick/place/sort/organize)?

If it fails any point → reject.
Output only the JSON as defined in the system prompt.
"""



# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """单张候选图片及其补充信息。"""

    identifier: str
    image_path: Path
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    identifier: str
    image_path: str
    decision: str
    score: float
    reason: str
    room_tags: List[str]
    object_tags: List[str]
    raw_response: Dict[str, Any]

    @property
    def keep(self) -> bool:
        return self.decision.lower() == "keep"


# ---------------------------------------------------------------------------
# 工具函数：输入输出
# ---------------------------------------------------------------------------


def load_candidates_from_manifest(manifest_path: Path) -> List[Candidate]:
    """读取 JSON/JSONL 清单，返回候选列表。"""
    data: List[Dict[str, Any]] = []
    text = manifest_path.read_text(encoding="utf-8")

    if manifest_path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line in {manifest_path}: {line[:80]}") from exc
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - 用户输入问题
            raise ValueError(f"Invalid JSON manifest {manifest_path}") from exc
        if isinstance(parsed, dict):
            # 允许 dict 映射：{"id": {...}, ...}
            data = [
                dict({"id": key}, **(value if isinstance(value, dict) else {"image_path": value}))
                for key, value in parsed.items()
            ]
        elif isinstance(parsed, list):
            data = [item for item in parsed if isinstance(item, dict)]
        else:  # pragma: no cover
            raise ValueError("Manifest must be JSON object or array of objects.")

    candidates: List[Candidate] = []
    for idx, item in enumerate(data):
        identifier = str(item.get("id") or item.get("identifier") or f"row-{idx:05d}")
        image_path = item.get("image_path") or item.get("image") or item.get("path")
        if not image_path:
            raise ValueError(f"Entry {identifier} missing 'image_path'.")
        candidates.append(
            Candidate(
                identifier=identifier,
                image_path=Path(image_path),
                extra={k: v for k, v in item.items() if k not in {"id", "identifier", "image_path", "image", "path"}},
            )
        )
    return candidates


def gather_candidates_from_dir(directory: Path) -> List[Candidate]:
    """扫描目录下所有图片，按照字典序排序。"""
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    files = [
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    files.sort()
    candidates = [
        Candidate(identifier=p.stem, image_path=p) for p in files
    ]
    return candidates


def ensure_dir(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 模型调用相关
# ---------------------------------------------------------------------------

_openai_client: Optional["OpenAI"] = None


def get_openai_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("未安装 openai 库，请先安装或改用 '--backend ollama-*'。")
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def encode_image_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def encode_image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    encoded = encode_image_to_base64(path)
    return f"data:{mime};base64,{encoded}"


def request_openai_vlm(model: str, image_url: str, timeout: int = 120) -> Dict[str, Any]:
    """调用 OpenAI Responses API，让模型返回 JSON。"""
    messages = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT.strip()}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": USER_PROMPT_TEMPLATE.strip()},
                {"type": "input_image", "image_url": image_url},
            ],
        },
    ]

    client = get_openai_client()
    response = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=400,
        timeout=timeout,
    )

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                return json.loads(text)
            value = getattr(text, "value", None) if text is not None else None
            if isinstance(value, str) and value.strip():
                return json.loads(value)

    alt_text = getattr(response, "output_text", None)
    if isinstance(alt_text, str) and alt_text.strip():
        return json.loads(alt_text)

    raise RuntimeError("Model response did not contain usable JSON.")


def request_ollama_vlm(
    *,
    model: str,
    image_base64: str,
    base_url: str,
    timeout: int,
    api_key: Optional[str],
) -> Dict[str, Any]:
    """调用 Ollama API（本地或云端），要求模型返回 JSON。"""
    import requests

    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.strip(),
                "images": [image_base64],
            },
        ],
        "stream": False,
    }
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(url, json=payload, headers=headers or None, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()

    message = body.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return json.loads(content)
    response_text = body.get("response")
    if isinstance(response_text, str) and response_text.strip():
        return json.loads(response_text)

    raise RuntimeError(f"Ollama response missing JSON content: {body}")


def jitter_sleep(base: float, jitter: float) -> None:
    time.sleep(base * (1.0 + random.uniform(-jitter, jitter)))


def call_vlm(
    candidate: Candidate,
    *,
    backend: str,
    model: str,
    timeout: int,
    ollama_url: Optional[str],
    ollama_api_key: Optional[str],
) -> Dict[str, Any]:
    """带重试地调用模型，兼容多个后端。"""
    last_error: Optional[Exception] = None
    data_url: Optional[str] = None
    image_base64: Optional[str] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if backend == "openai":
                if data_url is None:
                    data_url = encode_image_to_data_url(candidate.image_path)
                return request_openai_vlm(model=model, image_url=data_url, timeout=timeout)

            if backend in {"ollama-local", "ollama-cloud"}:
                if image_base64 is None:
                    image_base64 = encode_image_to_base64(candidate.image_path)
                if backend == "ollama-local":
                    base_url = ollama_url or DEFAULT_OLLAMA_LOCAL_URL
                    api_key = None
                else:
                    base_url = ollama_url or DEFAULT_OLLAMA_CLOUD_URL
                    api_key = ollama_api_key or os.environ.get(OLLAMA_API_KEY_ENV)
                    if not api_key:
                        raise RuntimeError(
                            "使用 Ollama 云端模型需要提供 API Key，请通过环境变量 "
                            f"{OLLAMA_API_KEY_ENV} 或 '--ollama-api-key' 传入。"
                        )
                return request_ollama_vlm(
                    model=model,
                    image_base64=image_base64,
                    base_url=base_url,
                    timeout=timeout,
                    api_key=api_key,
                )

            raise ValueError(f"Unsupported backend: {backend}")
        except Exception as exc:  # pragma: no cover - 依赖外部服务
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            backoff = RETRY_BACKOFF[0] * (1.8 ** (attempt - 1))
            jitter = RETRY_BACKOFF[1]
            jitter_sleep(backoff, jitter)
    raise RuntimeError(f"VLM call failed for {candidate.identifier}: {last_error}")


# ---------------------------------------------------------------------------
# 后处理
# ---------------------------------------------------------------------------


def normalise_string_list(values: Any, *, limit: int = 8) -> List[str]:
    if isinstance(values, str):
        parts = re.split(r"[;,/、|｜\n]+", values)
    elif isinstance(values, Sequence):
        parts = [str(v) for v in values]
    else:
        parts = []
    cleaned: List[str] = []
    seen = set()
    for part in parts:
        normalized = part.strip().lower()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
        if len(cleaned) >= limit:
            break
    return cleaned


def postprocess_response(candidate: Candidate, raw: Dict[str, Any]) -> Decision:
    decision = str(raw.get("decision") or "").strip().lower()
    if decision not in {"keep", "reject"}:
        decision = "reject"
    score = raw.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))

    reason = str(raw.get("reason") or "").strip()
    reason = reason[:160] if reason else ""

    room_tags = normalise_string_list(raw.get("room_tags"))
    object_tags = normalise_string_list(raw.get("object_tags"), limit=10)

    processed = {
        "decision": decision,
        "score": score,
        "reason": reason,
        "room_tags": room_tags,
        "object_tags": object_tags,
    }

    return Decision(
        identifier=candidate.identifier,
        image_path=str(candidate.image_path),
        decision=processed["decision"],
        score=processed["score"],
        reason=processed["reason"],
        room_tags=processed["room_tags"],
        object_tags=processed["object_tags"],
        raw_response={**raw, **processed},
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def process_candidate(
    candidate: Candidate,
    *,
    backend: str,
    model: str,
    timeout: int,
    ollama_url: Optional[str],
    ollama_api_key: Optional[str],
) -> Decision:
    raw = call_vlm(
        candidate,
        backend=backend,
        model=model,
        timeout=timeout,
        ollama_url=ollama_url,
        ollama_api_key=ollama_api_key,
    )
    return postprocess_response(candidate, raw)


def copy_if_needed(src: Path, dest_dir: Optional[Path]) -> Optional[Path]:
    if dest_dir is None:
        return None
    dest_path = dest_dir / src.name
    shutil.copy2(src, dest_path)
    return dest_path


def run(
    candidates: Sequence[Candidate],
    *,
    backend: str,
    model: str,
    timeout: int,
    workers: int,
    min_score: float,
    output_path: Path,
    keep_dir: Optional[Path],
    reject_dir: Optional[Path],
    ollama_url: Optional[str],
    ollama_api_key: Optional[str],
) -> List[Decision]:
    ensure_dir(keep_dir)
    ensure_dir(reject_dir)

    bar = tqdm(total=len(candidates), desc="VLM filtering", unit="img") if tqdm else None
    results: List[Decision] = []
    errors: Dict[str, str] = {}

    def _work(item: Candidate) -> Decision:
        return process_candidate(
            item,
            backend=backend,
            model=model,
            timeout=timeout,
            ollama_url=ollama_url,
            ollama_api_key=ollama_api_key,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_work, cand): cand for cand in candidates}
        for future in concurrent.futures.as_completed(future_map):
            candidate = future_map[future]
            try:
                decision = future.result()
                results.append(decision)
                keep = decision.keep and decision.score >= min_score
                if keep:
                    copy_if_needed(candidate.image_path, keep_dir)
                else:
                    copy_if_needed(candidate.image_path, reject_dir)
            except Exception as exc:
                errors[candidate.identifier] = repr(exc)
            finally:
                if bar:
                    bar.update(1)
                    if errors:
                        bar.set_postfix(err=len(errors))
    if bar:
        bar.close()

    results.sort(key=lambda d: d.identifier)
    write_jsonl(
        output_path,
        [
            {
                "id": d.identifier,
                "image_path": d.image_path,
                "decision": d.decision,
                "score": d.score,
                "reason": d.reason,
                "room_tags": d.room_tags,
                "object_tags": d.object_tags,
                "raw_response": d.raw_response,
            }
            for d in results
        ],
    )

    if errors:
        err_path = output_path.with_suffix(".errors.json")
        err_path.write_text(
            json.dumps({"count": len(errors), "errors": errors}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[WARN] {len(errors)} items failed. Details -> {err_path}")

    print(f"Saved {len(results)} records to {output_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use a vision-language model to filter indoor scenes.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-manifest", type=str, help="Path to JSON/JSONL manifest describing candidate images.")
    group.add_argument("-i", "--input-dir", type=str, help="Directory to recursively scan for images.")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="vlm_filter_results.jsonl",
        help="Output JSONL file (default: vlm_filter_results.jsonl).",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="VLM model name (default: %(default)s).")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout seconds.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Concurrent workers (default from WORKERS env).")
    parser.add_argument("--min-score", type=float, default=0.6, help="Minimum score to treat as keep.")
    parser.add_argument("--keep-dir", type=str, help="Optional directory to copy kept images.")
    parser.add_argument("--reject-dir", type=str, help="Optional directory to copy rejected images.")
    parser.add_argument(
        "--backend",
        choices=("openai", "ollama-local", "ollama-cloud"),
        default="openai",
        help="VLM backend to use (default: openai).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        help=f"Override Ollama base URL (defaults: local -> {DEFAULT_OLLAMA_LOCAL_URL}, cloud -> {DEFAULT_OLLAMA_CLOUD_URL}).",
    )
    parser.add_argument(
        "--ollama-api-key",
        type=str,
        help=f"API key for Ollama cloud backend (or set env {OLLAMA_API_KEY_ENV}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N candidates (after sorting). Useful for dry runs.",
    )
    parser.add_argument("--seed", type=int, help="Optional random seed for deterministic retries.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    if args.input_manifest:
        candidates = load_candidates_from_manifest(Path(args.input_manifest))
    else:
        candidates = gather_candidates_from_dir(Path(args.input_dir))

    if args.limit is not None:
        candidates = candidates[: args.limit]

    if not candidates:
        print("No candidates to process.", file=sys.stderr)
        return

    run(
        candidates,
        backend=args.backend,
        model=args.model,
        timeout=args.timeout,
        workers=max(1, args.workers),
        min_score=args.min_score,
        output_path=Path(args.output),
        keep_dir=Path(args.keep_dir) if args.keep_dir else None,
        reject_dir=Path(args.reject_dir) if args.reject_dir else None,
        ollama_url=args.ollama_url,
        ollama_api_key=args.ollama_api_key,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - 交互使用时友好退出
        print("\nInterrupted by user.", file=sys.stderr)
