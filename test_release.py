#!/usr/bin/env python3
"""Open-source evaluation entrypoint for MedVeriSeg verification.

Public single-sample input:
- original image path
- similarity heatmap path
- target class text
- similarity metrics S, C, and P
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from .qwen_api_medvriseg_release import (
        Config,
        DEFAULT_API_BASE_URL,
        QwenVisionLLM,
        RELEASE_MODEL_NAME,
        VerificationOutput,
        quantitative_response_tool_agent,
        verify_query_validity,
    )
except ImportError:
    from qwen_api_medvriseg_release import (
        Config,
        DEFAULT_API_BASE_URL,
        QwenVisionLLM,
        RELEASE_MODEL_NAME,
        VerificationOutput,
        quantitative_response_tool_agent,
        verify_query_validity,
    )


S_THRESHOLD = 0.475
C_THRESHOLD = 0.4
P_THRESHOLD = 0.7
FINAL_DECISION_THRESHOLD = 0.5
PROGRESS_EVERY = 50
MODEL_LIST = [RELEASE_MODEL_NAME]
MODEL_FAIL_SWITCH_THRESHOLD = 5

_QWEN_LLM: Optional[QwenVisionLLM] = None


def load_item_json(item_dir: Path) -> Dict[str, Any]:
    data_path = item_dir / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.json: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_original_image(item_dir: Path) -> Path:
    candidates = sorted(item_dir.glob("original_image.*"))
    if not candidates:
        raise FileNotFoundError(f"Missing original_image.* in {item_dir}")
    return candidates[0]


def make_config(api_base_url: str = DEFAULT_API_BASE_URL) -> Config:
    return Config(
        tau_s=S_THRESHOLD,
        tau_c=C_THRESHOLD,
        tau_p=P_THRESHOLD,
        theta=FINAL_DECISION_THRESHOLD,
        api_model=MODEL_LIST[0],
        api_models=MODEL_LIST,
        api_base_url=api_base_url,
        temperature=0.0,
        max_new_tokens=512,
        max_json_retries=5,
        model_fail_switch_threshold=MODEL_FAIL_SWITCH_THRESHOLD,
    )


def get_model_state() -> Dict[str, Any]:
    if _QWEN_LLM is None:
        return {
            "current_model": None,
            "api_models": list(MODEL_LIST),
            "model_fail_switch_threshold": MODEL_FAIL_SWITCH_THRESHOLD,
            "switch_count": 0,
            "switch_history": [],
        }
    return _QWEN_LLM.model_state()


def run_agent(
    class_text: str,
    original_image_path: Path,
    heatmap_path: Path,
    S: float,
    C: float,
    P: float,
    api_base_url: str = DEFAULT_API_BASE_URL,
):
    """
    Run the original rule-plus-agent verification logic for one sample.

    Args:
        class_text: Target class or anatomy text used in the segmentation query.
        original_image_path: Path to the original medical image.
        heatmap_path: Path to the similarity heatmap or heatmap overlay image.
        S: Similarity strength metric.
        C: Similarity compactness metric.
        P: Similarity purity metric.
        api_base_url: OpenAI-compatible endpoint URL.
    """
    global _QWEN_LLM

    cfg = make_config(api_base_url=api_base_url)
    query = f"Please segment the {class_text} in the medical image."

    prediction, source = rule_decision({"S": S, "C": C, "P": P})
    if prediction is not None:
        quantitative_card = quantitative_response_tool_agent(S, C, P, cfg)
        final_decision = "present" if prediction else "absent"
        return VerificationOutput(
            route=source,
            final_decision=final_decision,
            final_confidence=quantitative_card.confidence,
            existence_score=None,
            reason=(
                "The script-level quantitative rule made the final decision. "
                "No visual agent was activated."
            ),
            evidence_cards={
                "quantitative": asdict(quantitative_card),
            },
            weights=None,
            raw_outputs={},
        )

    if _QWEN_LLM is None:
        _QWEN_LLM = QwenVisionLLM(
            api_model=cfg.api_model,
            api_models=cfg.api_models,
            api_base_url=cfg.api_base_url,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            max_json_retries=cfg.max_json_retries,
            model_fail_switch_threshold=cfg.model_fail_switch_threshold,
        )

    return verify_query_validity(
        query=query,
        original_image_path=str(original_image_path),
        heatmap_path=str(heatmap_path),
        S=float(S),
        C=float(C),
        P=float(P),
        cfg=cfg,
        llm=_QWEN_LLM,
    )


def agent_decision(
    class_text: str,
    original_image_path: Path,
    heatmap_path: Path,
    S: float,
    C: float,
    P: float,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> bool:
    result = run_agent(
        class_text=class_text,
        original_image_path=original_image_path,
        heatmap_path=heatmap_path,
        S=S,
        C=C,
        P=P,
        api_base_url=api_base_url,
    )
    return result.final_decision == "present"


def record_path(record_dir: Path, split_name: str, item_dir: Path) -> Path:
    return record_dir / split_name / f"{item_dir.name}.json"


def load_completed_record(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)
    except Exception as exc:
        print(f"record: cannot read {path}, will rerun item. error={exc}", flush=True)
        return None
    if record.get("status") != "completed":
        return None
    return record


def write_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def build_record(
    split_name: str,
    item_dir: Path,
    truth: bool,
    prediction: bool,
    source: str,
    used_agent: bool,
    class_text: str,
    metrics: Dict[str, Any],
    agent_result: Any = None,
) -> Dict[str, Any]:
    return {
        "status": "completed",
        "split": split_name,
        "item": item_dir.name,
        "truth": truth,
        "prediction": prediction,
        "correct": prediction == truth,
        "used_agent": used_agent,
        "source": source,
        "class_text": class_text,
        "similarity_metrics": metrics,
        "agent_result": asdict(agent_result) if agent_result is not None else None,
        "model_state": get_model_state(),
    }


def rule_decision(metrics: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    s = float(metrics["S"])
    c = float(metrics["C"])
    p = float(metrics["P"])

    if c >= C_THRESHOLD and p >= P_THRESHOLD and s >= S_THRESHOLD:
        return True, "rule_true"
    if c < C_THRESHOLD and p < P_THRESHOLD and s < S_THRESHOLD:
        return False, "rule_false"
    return None, "agent"


def iter_item_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Missing result directory: {root}")
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("item_")),
        key=lambda path: int(path.name.split("_")[-1]),
    )


def evaluate_split(
    root: Path,
    truth: bool,
    split_name: str,
    record_dir: Path,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> Dict[str, Any]:
    item_dirs = list(iter_item_dirs(root))
    total_items = len(item_dirs)
    total = 0
    correct = 0
    rule_true_count = 0
    rule_false_count = 0
    agent_count = 0
    errors: List[str] = []

    print(f"{split_name}: starting evaluation, total={total_items}", flush=True)

    for idx, item_dir in enumerate(item_dirs, start=1):
        item_record_path = record_path(record_dir, split_name, item_dir)
        existing_record = load_completed_record(item_record_path)
        total += 1

        if existing_record is not None:
            prediction = bool(existing_record["prediction"])
            source = existing_record.get("source", "unknown")
            used_agent = bool(existing_record.get("used_agent", False))

            if source == "rule_true":
                rule_true_count += 1
            elif source == "rule_false":
                rule_false_count += 1
            elif used_agent or source == "agent":
                agent_count += 1

            if prediction == truth:
                correct += 1
            else:
                errors.append(item_dir.name)

            print(
                f"{split_name}: [{idx}/{total_items}] skip recorded {item_dir.name} "
                f"source={source} prediction={prediction}",
                flush=True,
            )
            if idx == 1 or idx % PROGRESS_EVERY == 0 or idx == total_items:
                accuracy_so_far = correct / total if total else 0.0
                print(
                    f"{split_name}: progress {idx}/{total_items} "
                    f"correct={correct} wrong={total - correct} "
                    f"acc={accuracy_so_far:.6f} rule_true={rule_true_count} "
                    f"rule_false={rule_false_count} agent={agent_count}",
                    flush=True,
                )
            continue

        data = load_item_json(item_dir)
        metrics = data["similarity_metrics"]
        original_item = data["original_jsonl_item"]
        class_text = original_item["class_text"]
        used_agent = False
        agent_result = None
        prediction, source = rule_decision(metrics)

        if prediction is True and source == "rule_true":
            rule_true_count += 1
        elif prediction is False and source == "rule_false":
            rule_false_count += 1
        else:
            used_agent = True
            agent_count += 1
            original_image_path = find_original_image(item_dir)
            heatmap_path = item_dir / "similarity_map.png"
            print(
                f"{split_name}: [{idx}/{total_items}] calling agent for "
                f"{item_dir.name}, class={class_text!r}",
                flush=True,
            )
            agent_result = run_agent(
                class_text=class_text,
                original_image_path=original_image_path,
                heatmap_path=heatmap_path,
                S=float(metrics["S"]),
                C=float(metrics["C"]),
                P=float(metrics["P"]),
                api_base_url=api_base_url,
            )
            prediction = agent_result.final_decision == "present"

        if prediction == truth:
            correct += 1
        else:
            errors.append(item_dir.name)

        write_record(
            item_record_path,
            build_record(
                split_name=split_name,
                item_dir=item_dir,
                truth=truth,
                prediction=prediction,
                source=source,
                used_agent=used_agent,
                class_text=class_text,
                metrics=metrics,
                agent_result=agent_result,
            ),
        )

        if idx == 1 or idx % PROGRESS_EVERY == 0 or idx == total_items:
            accuracy_so_far = correct / total if total else 0.0
            print(
                f"{split_name}: progress {idx}/{total_items} "
                f"correct={correct} wrong={total - correct} "
                f"acc={accuracy_so_far:.6f} rule_true={rule_true_count} "
                f"rule_false={rule_false_count} agent={agent_count}",
                flush=True,
            )

    accuracy = correct / total if total else 0.0
    return {
        "split": split_name,
        "truth": truth,
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": accuracy,
        "rule_true": rule_true_count,
        "rule_false": rule_false_count,
        "agent": agent_count,
        "wrong_items": errors,
    }


def print_result(result: Dict[str, Any]) -> None:
    print(
        f"{result['split']}: total={result['total']} correct={result['correct']} "
        f"wrong={result['wrong']} acc={result['accuracy']:.6f} "
        f"rule_true={result['rule_true']} rule_false={result['rule_false']} "
        f"agent={result['agent']}"
    )
    if result["wrong_items"]:
        preview = ", ".join(result["wrong_items"][:20])
        extra = len(result["wrong_items"]) - 20
        suffix = "" if extra <= 0 else f", ... (+{extra})"
        print(f"{result['split']} wrong_items: {preview}{suffix}")


def run_single_sample(args: argparse.Namespace) -> Dict[str, Any]:
    result = run_agent(
        class_text=args.class_text,
        original_image_path=args.original_image,
        heatmap_path=args.heatmap,
        S=args.S,
        C=args.C,
        P=args.P,
        api_base_url=args.api_base_url,
    )
    output = asdict(result)
    output["model_state"] = get_model_state()
    return output


def run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    negative_dir = args.root / "negative"
    positive_dir = args.root / "positive"
    record_dir = args.record_dir or (args.root / "record")

    negative_result = evaluate_split(
        negative_dir,
        truth=False,
        split_name="negative",
        record_dir=record_dir,
        api_base_url=args.api_base_url,
    )
    positive_result = evaluate_split(
        positive_dir,
        truth=True,
        split_name="positive",
        record_dir=record_dir,
        api_base_url=args.api_base_url,
    )

    print_result(negative_result)
    print_result(positive_result)

    total = negative_result["total"] + positive_result["total"]
    correct = negative_result["correct"] + positive_result["correct"]
    accuracy = correct / total if total else 0.0
    overall = {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": accuracy,
    }
    print(
        f"overall: total={total} correct={correct} "
        f"wrong={total - correct} acc={accuracy:.6f}"
    )
    return {
        "negative": negative_result,
        "positive": positive_result,
        "overall": overall,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MedVeriSeg verification with original image, heatmap, "
            "class text, and similarity metrics S/C/P."
        )
    )
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help="OpenAI-compatible API base URL. Defaults to OPENAI_BASE_URL or a placeholder URL.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for writing the JSON result.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--root",
        type=Path,
        help=(
            "Optional batch root containing negative/ and positive/ item directories. "
            "Each item must contain data.json, original_image.*, and similarity_map.png."
        ),
    )
    mode.add_argument(
        "--original-image",
        type=Path,
        help="Path to the original image for single-sample mode.",
    )

    parser.add_argument(
        "--heatmap",
        type=Path,
        help="Path to the similarity heatmap for single-sample mode.",
    )
    parser.add_argument(
        "--class-text",
        help="Target class text for single-sample mode.",
    )
    parser.add_argument("--S", type=float, help="Similarity strength metric.")
    parser.add_argument("--C", type=float, help="Similarity compactness metric.")
    parser.add_argument("--P", type=float, help="Similarity purity metric.")
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help="Batch-mode record directory. Defaults to <root>/record.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.root is not None:
        return

    missing = []
    for name in ["heatmap", "class_text", "S", "C", "P"]:
        if getattr(args, name) is None:
            missing.append("--" + name.replace("_", "-"))
    if missing:
        raise SystemExit(
            "Single-sample mode requires: "
            "--original-image, --heatmap, --class-text, --S, --C, and --P. "
            f"Missing: {', '.join(missing)}"
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    if args.root is not None:
        result = run_batch(args)
    else:
        result = run_single_sample(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
