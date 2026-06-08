"""
Lightweight routed multi-agent verification for MedVeriSeg.

This release version uses an OpenAI-compatible Qwen-3-VL endpoint. It never
stores an API key in code; set OPENAI_API_KEY in the environment before making
API calls.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    from typing import Literal
except ImportError:
    class _LiteralFallback:
        def __getitem__(self, item: Any) -> Any:
            return Any

    Literal = _LiteralFallback()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


Decision = Literal["present", "absent", "uncertain"]
FinalDecision = Literal["present", "absent"]

DEFAULT_API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.example.com/v1")
RELEASE_MODEL_NAME = "Qwen-3-VL"


@dataclass
class Config:
    tau_s: float = 0.60
    tau_c: float = 0.50
    tau_p: float = 0.60
    theta: float = 0.50

    api_model: str = RELEASE_MODEL_NAME
    api_models: Optional[List[str]] = None
    api_base_url: str = DEFAULT_API_BASE_URL

    max_new_tokens: int = 512
    temperature: float = 0.0
    max_json_retries: int = 5
    model_fail_switch_threshold: int = 5


@dataclass
class EvidenceCard:
    agent: str
    decision: Decision
    confidence: float
    evidence: Dict[str, Any]
    risk: str = ""


@dataclass
class VerificationOutput:
    route: str
    final_decision: FinalDecision
    final_confidence: float
    existence_score: Optional[float]
    reason: str
    evidence_cards: Dict[str, Any]
    weights: Optional[Dict[str, float]]
    raw_outputs: Dict[str, Any]


def clip01(x: Any, default: float = 0.5) -> float:
    try:
        x = float(x)
    except Exception:
        x = default
    return max(0.0, min(1.0, x))


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"Cannot parse JSON from model output:\n{text}")


def normalize_weights(weights: Dict[str, Any]) -> Dict[str, float]:
    keys = ["quantitative", "raw_image", "heatmap"]
    clean: Dict[str, float] = {}

    for key in keys:
        try:
            clean[key] = max(0.0, float(weights.get(key, 0.0) or 0.0))
        except Exception:
            clean[key] = 0.0

    total = sum(clean.values())
    if total <= 1e-8:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: clean[key] / total for key in keys}


def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_data_url(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "image/jpeg"
    return f"data:{mime_type};base64,{b64_image(path)}"


def response_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def usage_to_dict(usage: Any) -> Dict[str, int]:
    values: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if usage is None:
        return values

    for key in values:
        if isinstance(usage, dict):
            raw_value = usage.get(key, 0)
        else:
            raw_value = getattr(usage, key, 0)
        try:
            values[key] = int(raw_value or 0)
        except Exception:
            values[key] = 0
    return values


def empty_token_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


class QwenVisionLLM:
    """
    OpenAI-compatible Qwen-3-VL wrapper.

    Args:
        api_model: Public model identifier. The release default is
            ``Qwen-3-VL``.
        api_models: Optional fallback model list. In the release version, use
            sanitized public names only.
        api_base_url: OpenAI-compatible API base URL.
        temperature: Sampling temperature. Default: 0.0.
        max_new_tokens: Maximum output tokens. Default: 512.
        max_json_retries: Number of attempts to obtain parseable JSON.
            Default: 5.
        model_fail_switch_threshold: Consecutive failures before trying the
            next model in ``api_models``. Default: 5.
        client: Optional injected client for tests or custom runtimes.
    """

    def __init__(
        self,
        api_model: str = RELEASE_MODEL_NAME,
        api_models: Optional[List[str]] = None,
        api_base_url: str = DEFAULT_API_BASE_URL,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        max_json_retries: int = 5,
        model_fail_switch_threshold: int = 5,
        client: Optional[Any] = None,
    ):
        self.api_models = list(api_models or [api_model])
        if not self.api_models:
            raise ValueError("api_models must contain at least one model.")
        self.current_model_index = 0
        self.api_model = self.api_models[self.current_model_index]
        self.api_base_url = api_base_url
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_json_retries = max(1, int(max_json_retries))
        self.model_fail_switch_threshold = max(1, int(model_fail_switch_threshold))
        self.model_failure_counts = {model: 0 for model in self.api_models}
        self.model_switch_history: List[Dict[str, Any]] = []
        self.total_usage = empty_token_usage()
        self.api_call_log: List[Dict[str, Any]] = []

        if client is None:
            if OpenAI is None:
                raise ImportError(
                    "The openai package is required for API calls. "
                    "Install it with `pip install openai`."
                )
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set before calling the API.")
            self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        else:
            self.client = client

    @property
    def current_model(self) -> str:
        return self.api_models[self.current_model_index]

    def model_state(self) -> Dict[str, Any]:
        return {
            "current_model": self.current_model,
            "api_models": list(self.api_models),
            "model_failure_counts": dict(self.model_failure_counts),
            "model_fail_switch_threshold": self.model_fail_switch_threshold,
            "switch_count": len(self.model_switch_history),
            "switch_history": list(self.model_switch_history),
        }

    def usage_state(self) -> Dict[str, Any]:
        return {
            "token_usage": dict(self.total_usage),
            "api_call_count": len(self.api_call_log),
            "api_calls": [dict(call) for call in self.api_call_log],
        }

    def _record_api_call(
        self,
        model: str,
        attempt: int,
        elapsed_seconds: float,
        usage: Any,
        parse_success: bool,
    ) -> None:
        usage_dict = usage_to_dict(usage)
        for key, value in usage_dict.items():
            self.total_usage[key] += value
        self.api_call_log.append(
            {
                "model": model,
                "attempt": attempt,
                "elapsed_seconds": elapsed_seconds,
                "usage": usage_dict,
                "parse_success": parse_success,
            }
        )

    def _register_model_success(self, model: str) -> None:
        self.model_failure_counts[model] = 0
        self.api_model = model

    def _register_model_failure(self, model: str, error: Exception) -> None:
        self.model_failure_counts[model] = self.model_failure_counts.get(model, 0) + 1
        failure_count = self.model_failure_counts[model]
        if failure_count < self.model_fail_switch_threshold:
            return

        old_model = model
        old_index = self.api_models.index(old_model)
        for offset in range(1, len(self.api_models) + 1):
            next_index = (old_index + offset) % len(self.api_models)
            next_model = self.api_models[next_index]
            if self.model_failure_counts.get(next_model, 0) < self.model_fail_switch_threshold:
                self.current_model_index = next_index
                self.api_model = next_model
                switch_event = {
                    "from": old_model,
                    "to": next_model,
                    "failure_count": failure_count,
                    "reason": str(error),
                }
                self.model_switch_history.append(switch_event)
                print(
                    f"Qwen-3-VL API model switch: {old_model} -> {next_model} "
                    f"after {failure_count} consecutive failure(s).",
                    flush=True,
                )
                return

        raise RuntimeError(
            "All Qwen-3-VL API models failed. "
            f"models={self.api_models}; "
            f"failure_counts={self.model_failure_counts}; "
            f"last_error={error}"
        )

    def json_call(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []

        if image_paths:
            for path in image_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image not found: {path}")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(path)},
                    }
                )

        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        last_error: Optional[Exception] = None

        while True:
            model = self.current_model
            last_output_text = ""
            parse_error: Optional[Exception] = None

            try:
                for attempt in range(1, self.max_json_retries + 1):
                    call_started_at = time.perf_counter()
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_new_tokens,
                    )
                    call_elapsed = time.perf_counter() - call_started_at
                    last_output_text = response_content_to_text(
                        resp.choices[0].message.content
                    )

                    try:
                        parsed = extract_json(last_output_text)
                        self._record_api_call(
                            model=model,
                            attempt=attempt,
                            elapsed_seconds=call_elapsed,
                            usage=getattr(resp, "usage", None),
                            parse_success=True,
                        )
                        self._register_model_success(model)
                        return parsed
                    except Exception as exc:
                        self._record_api_call(
                            model=model,
                            attempt=attempt,
                            elapsed_seconds=call_elapsed,
                            usage=getattr(resp, "usage", None),
                            parse_success=False,
                        )
                        parse_error = exc

                last_error = ValueError(
                    f"Cannot parse valid JSON from Qwen-3-VL API output after "
                    f"{self.max_json_retries} attempt(s) with model {model}. "
                    f"Last parse error: {parse_error}\n"
                    f"Last model output:\n{last_output_text}"
                )
            except Exception as exc:
                last_error = exc

            self._register_model_failure(model, last_error)


def quantitative_response_tool_agent(
    S: float,
    C: float,
    P: float,
    cfg: Config,
) -> EvidenceCard:
    scores = {
        "strength": float(S),
        "compactness": float(C),
        "purity": float(P),
    }
    thresholds = {
        "strength": cfg.tau_s,
        "compactness": cfg.tau_c,
        "purity": cfg.tau_p,
    }
    states = {}
    margins = {}

    for key in scores:
        margin = scores[key] - thresholds[key]
        margins[key] = margin
        if scores[key] > thresholds[key]:
            states[key] = "above_threshold"
        elif scores[key] < thresholds[key]:
            states[key] = "below_threshold"
        else:
            states[key] = "on_threshold"

    n_above = sum(value == "above_threshold" for value in states.values())
    n_below = sum(value == "below_threshold" for value in states.values())

    if n_above == 3:
        decision: Decision = "present"
    elif n_below == 3:
        decision = "absent"
    else:
        decision = "uncertain"

    confidence = clip01(sum(abs(value) for value in margins.values()) / 3.0)
    if decision == "uncertain":
        confidence = min(confidence, 0.65)

    evidence = {
        "scores": scores,
        "thresholds": thresholds,
        "states": states,
        "margins": margins,
        "summary": (
            f"S={S:.4f}, C={C:.4f}, P={P:.4f}; "
            f"thresholds=({cfg.tau_s:.4f}, {cfg.tau_c:.4f}, {cfg.tau_p:.4f})"
        ),
    }

    if decision == "present":
        risk = "All quantitative indicators support target presence."
    elif decision == "absent":
        risk = "All quantitative indicators support target absence."
    else:
        risk = "Quantitative indicators are inconsistent; additional visual evidence is required."

    return EvidenceCard(
        agent="Quantitative Response Tool Agent",
        decision=decision,
        confidence=confidence,
        evidence=evidence,
        risk=risk,
    )


def dynamic_router(S: float, C: float, P: float, cfg: Config) -> str:
    all_above = (S > cfg.tau_s) and (C > cfg.tau_c) and (P > cfg.tau_p)
    all_below = (S < cfg.tau_s) and (C < cfg.tau_c) and (P < cfg.tau_p)

    if all_above:
        return "direct_present"
    if all_below:
        return "direct_absent"
    return "agent_verification"


def raw_image_semantic_agent(
    llm: QwenVisionLLM,
    query: str,
    original_image_path: str,
) -> EvidenceCard:
    prompt = f"""
You are a Raw Image Semantic Agent for medical segmentation query verification.

Input:
- Query: {query}
- Original medical image is provided.

Task:
Judge whether the original medical image provides visible or plausible evidence that the queried target exists.

Important rules:
1. Do not perform segmentation.
2. Do not use heatmap information.
3. Do not overclaim. If the target cannot be reliably determined from the raw image alone, choose "uncertain".
4. Your decision must be one of: "present", "absent", "uncertain".
5. Return JSON only. Do not include markdown fences. Do not include explanations outside JSON.

Return format:
{{
  "decision": "present/absent/uncertain",
  "confidence": 0.0,
  "evidence": [
    "brief visual evidence"
  ],
  "image_quality": "clear/low_contrast/ambiguous/small_target/unknown",
  "risk": "brief risk analysis"
}}
"""
    out = llm.json_call(prompt, [original_image_path])

    decision = out.get("decision", "uncertain")
    if decision not in ["present", "absent", "uncertain"]:
        decision = "uncertain"

    return EvidenceCard(
        agent="Raw Image Semantic Agent",
        decision=decision,
        confidence=clip01(out.get("confidence", 0.5)),
        evidence={
            "visual_evidence": out.get("evidence", []),
            "image_quality": out.get("image_quality", "unknown"),
        },
        risk=out.get("risk", ""),
    )


def heatmap_localization_agent(
    llm: QwenVisionLLM,
    query: str,
    heatmap_path: str,
) -> EvidenceCard:
    prompt = f"""
You are a Heatmap Localization Agent for medical segmentation query verification.

Input:
- Query: {query}
- Similarity heatmap or heatmap-overlay image is provided.

Task:
Judge whether the heatmap pattern supports the existence of the queried target.

Focus on:
1. whether the response is spatially concentrated,
2. whether there is a clear main peak,
3. whether the response is scattered or background-dominated,
4. whether the high-response region plausibly aligns with a target region,
5. whether the heatmap looks off-target or noisy.

Important rules:
1. Only evaluate the heatmap pattern.
2. Do not rely on raw image semantic appearance.
3. If the heatmap is not clearly supportive or clearly negative, choose "uncertain".
4. Your decision must be one of: "present", "absent", "uncertain".
5. Return JSON only. Do not include markdown fences. Do not include explanations outside JSON.

Return format:
{{
  "decision": "present/absent/uncertain",
  "confidence": 0.0,
  "localization_pattern": "focused/scattered/off_target/multi_peak/ambiguous",
  "evidence": [
    "brief heatmap evidence"
  ],
  "risk": "brief risk analysis"
}}
"""
    out = llm.json_call(prompt, [heatmap_path])

    decision = out.get("decision", "uncertain")
    if decision not in ["present", "absent", "uncertain"]:
        decision = "uncertain"

    return EvidenceCard(
        agent="Heatmap Localization Agent",
        decision=decision,
        confidence=clip01(out.get("confidence", 0.5)),
        evidence={
            "localization_pattern": out.get("localization_pattern", "ambiguous"),
            "heatmap_evidence": out.get("evidence", []),
        },
        risk=out.get("risk", ""),
    )


def evidence_weighting_agent(
    llm: QwenVisionLLM,
    query: str,
    quantitative_card: EvidenceCard,
    raw_image_card: EvidenceCard,
    heatmap_card: EvidenceCard,
) -> Dict[str, Any]:
    evidence_cards = {
        "quantitative": asdict(quantitative_card),
        "raw_image": asdict(raw_image_card),
        "heatmap": asdict(heatmap_card),
    }

    prompt = f"""
You are an Evidence Weighting Agent for medical segmentation query verification.

Query:
{query}

You will receive three evidence cards:
1. Quantitative Response Tool Agent
2. Raw Image Semantic Agent
3. Heatmap Localization Agent

Your task:
Assign reliability weights to the three evidence sources.

Important rules:
1. Do NOT output the final present/absent decision.
2. Only assign weights according to evidence reliability.
3. The weights must sum to 1.
4. Give higher weight to an evidence source if it is confident, specific, and well supported.
5. Give lower weight to an evidence source if it is uncertain, generic, or potentially unreliable.
6. If quantitative evidence is decisive, it can receive higher weight.
7. If quantitative evidence is inconsistent, visual evidence should receive higher weight.
8. If heatmap evidence is focused but raw image evidence is incompatible, reduce heatmap weight.
9. Return JSON only. Do not include markdown fences. Do not include explanations outside JSON.

Evidence cards:
{json.dumps(evidence_cards, indent=2, ensure_ascii=False)}

Return format:
{{
  "weights": {{
    "quantitative": 0.0,
    "raw_image": 0.0,
    "heatmap": 0.0
  }},
  "conflict_level": "none/mild/strong",
  "reason": "brief explanation for the weight assignment"
}}
"""
    out = llm.json_call(prompt, image_paths=None)
    weights = normalize_weights(out.get("weights", {}))

    return {
        "weights": weights,
        "conflict_level": out.get("conflict_level", "unknown"),
        "reason": out.get("reason", ""),
        "raw": out,
    }


def evidence_value(card: EvidenceCard) -> float:
    confidence = clip01(card.confidence)
    if card.decision == "present":
        return 0.5 + 0.5 * confidence
    if card.decision == "absent":
        return 0.5 - 0.5 * confidence
    return 0.5


def score_based_decider(
    quantitative_card: EvidenceCard,
    raw_image_card: EvidenceCard,
    heatmap_card: EvidenceCard,
    weights: Dict[str, float],
    theta: float,
) -> Dict[str, Any]:
    weights = normalize_weights(weights)

    x_q = evidence_value(quantitative_card)
    x_r = evidence_value(raw_image_card)
    x_h = evidence_value(heatmap_card)

    existence_score = (
        weights["quantitative"] * x_q
        + weights["raw_image"] * x_r
        + weights["heatmap"] * x_h
    )

    final_decision: FinalDecision = "present" if existence_score > theta else "absent"
    denom = max(theta, 1.0 - theta, 1e-8)
    final_confidence = clip01(abs(existence_score - theta) / denom)

    return {
        "final_decision": final_decision,
        "final_confidence": final_confidence,
        "existence_score": existence_score,
        "theta": theta,
        "evidence_values": {
            "quantitative": x_q,
            "raw_image": x_r,
            "heatmap": x_h,
        },
        "weights": weights,
    }


def verify_query_validity(
    query: str,
    original_image_path: str,
    heatmap_path: str,
    S: float,
    C: float,
    P: float,
    cfg: Optional[Config] = None,
    llm: Optional[QwenVisionLLM] = None,
) -> VerificationOutput:
    """
    Verify whether a segmentation query target is present.

    Inputs:
        query: Natural-language segmentation request.
        original_image_path: Path to the original image.
        heatmap_path: Path to the similarity heatmap or heatmap overlay.
        S: Similarity strength score.
        C: Similarity compactness score.
        P: Similarity purity score.
        cfg: Optional thresholds and API settings.
        llm: Optional initialized QwenVisionLLM instance.
    """
    if cfg is None:
        cfg = Config()

    quantitative_card = quantitative_response_tool_agent(S, C, P, cfg)
    route = dynamic_router(S, C, P, cfg)

    if route == "direct_present":
        return VerificationOutput(
            route=route,
            final_decision="present",
            final_confidence=quantitative_card.confidence,
            existence_score=None,
            reason="All three quantitative indicators are above their thresholds. No visual agent is activated.",
            evidence_cards={
                "quantitative": asdict(quantitative_card),
            },
            weights=None,
            raw_outputs={},
        )

    if route == "direct_absent":
        return VerificationOutput(
            route=route,
            final_decision="absent",
            final_confidence=quantitative_card.confidence,
            existence_score=None,
            reason="All three quantitative indicators are below their thresholds. No visual agent is activated.",
            evidence_cards={
                "quantitative": asdict(quantitative_card),
            },
            weights=None,
            raw_outputs={},
        )

    if llm is None:
        llm = QwenVisionLLM(
            api_model=cfg.api_model,
            api_models=cfg.api_models,
            api_base_url=cfg.api_base_url,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            max_json_retries=cfg.max_json_retries,
            model_fail_switch_threshold=cfg.model_fail_switch_threshold,
        )

    raw_image_card = raw_image_semantic_agent(
        llm=llm,
        query=query,
        original_image_path=original_image_path,
    )
    heatmap_card = heatmap_localization_agent(
        llm=llm,
        query=query,
        heatmap_path=heatmap_path,
    )
    weighting = evidence_weighting_agent(
        llm=llm,
        query=query,
        quantitative_card=quantitative_card,
        raw_image_card=raw_image_card,
        heatmap_card=heatmap_card,
    )
    score_result = score_based_decider(
        quantitative_card=quantitative_card,
        raw_image_card=raw_image_card,
        heatmap_card=heatmap_card,
        weights=weighting["weights"],
        theta=cfg.theta,
    )

    reason = (
        f"Final decision is computed by score-based evidence fusion. "
        f"ExistenceScore={score_result['existence_score']:.4f}, "
        f"theta={score_result['theta']:.4f}. "
        f"Weighting reason: {weighting.get('reason', '')}"
    )

    return VerificationOutput(
        route=route,
        final_decision=score_result["final_decision"],
        final_confidence=score_result["final_confidence"],
        existence_score=score_result["existence_score"],
        reason=reason,
        evidence_cards={
            "quantitative": asdict(quantitative_card),
            "raw_image": asdict(raw_image_card),
            "heatmap": asdict(heatmap_card),
        },
        weights=score_result["weights"],
        raw_outputs={
            "weighting": weighting,
            "score_result": score_result,
        },
    )
