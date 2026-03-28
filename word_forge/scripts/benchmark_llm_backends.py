#!/usr/bin/env python3
"""Benchmark Word Forge LLM backends on small structured lexical tasks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parents[1].resolve() / "src"))
sys.path.insert(0, str(Path(__file__).parents[2].resolve() / "lib"))

from word_forge.parser.language_model import ModelState, default_word_forge_model_name


@dataclass(frozen=True)
class PromptCase:
    name: str
    prompt: str
    expects_json: bool = False
    required_keys: tuple[str, ...] = ()


PROMPT_CASES: tuple[PromptCase, ...] = (
    PromptCase(
        name="definition",
        prompt="Define the word 'serendipity' in one concise sentence.",
    ),
    PromptCase(
        name="term_extraction",
        prompt=(
            "Task: extract salient lexical candidates from the sentence. "
            "Return only valid JSON using exactly this schema: "
            '{"phrases":[string],"entities":[string]}. '
            'Sentence: "The Atlas dashboard lets Eidos inspect Code Forge provenance and '
            'Word Forge graphs from a phone browser."'
        ),
        expects_json=True,
        required_keys=("phrases", "entities"),
    ),
    PromptCase(
        name="g2p",
        prompt=(
            'Generate pronunciations for the word "serendipity". '
            'Return only valid JSON using exactly this schema: '
            '{"ipa":"string","arpabet":"string","stress_pattern":"string"}'
        ),
        expects_json=True,
        required_keys=("ipa", "arpabet", "stress_pattern"),
    ),
)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
    return candidates[-1] if candidates else None


def score_case(case: PromptCase, output: Optional[str], latency_s: float) -> Dict[str, Any]:
    output_text = (output or "").strip()
    json_payload = extract_json_object(output_text) if case.expects_json else None
    keys_ok = bool(json_payload and all(key in json_payload for key in case.required_keys))
    valid = bool(output_text) and (keys_ok if case.expects_json else True)
    score = 1.0 if valid else 0.0
    if valid and latency_s > 20.0:
        score = 0.9
    if valid and latency_s > 40.0:
        score = 0.8
    return {
        "name": case.name,
        "latency_s": round(latency_s, 3),
        "output": output_text,
        "json_payload": json_payload,
        "valid": valid,
        "score": score,
    }


def benchmark_model(model_name: str, max_new_tokens: int = 96) -> Dict[str, Any]:
    state = ModelState(model_name=model_name)
    initialized = state.initialize()
    results: List[Dict[str, Any]] = []
    if not initialized:
        return {
            "model": model_name,
            "initialized": False,
            "score": 0.0,
            "cases": results,
        }

    for case in PROMPT_CASES:
        started = time.perf_counter()
        output = state.generate_text(case.prompt, max_new_tokens=max_new_tokens, temperature=0.2)
        latency_s = time.perf_counter() - started
        results.append(score_case(case, output, latency_s))

    overall = round(sum(item["score"] for item in results) / len(results), 3) if results else 0.0
    return {
        "model": model_name,
        "initialized": True,
        "score": overall,
        "total_latency_s": round(sum(item["latency_s"] for item in results), 3),
        "cases": results,
    }


def recommend_model(models: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    viable = [model for model in models if model.get("initialized")]
    if not viable:
        return None
    ranked = sorted(
        viable,
        key=lambda item: (
            -float(item.get("score", 0.0)),
            float(item.get("total_latency_s", 10_000.0)),
            item.get("model", ""),
        ),
    )
    winner = ranked[0]
    return {
        "model": winner["model"],
        "score": winner["score"],
        "total_latency_s": winner.get("total_latency_s"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Word Forge LLM backends")
    parser.add_argument("--model", action="append", dest="models", help="Model override; may be repeated")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON report")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max generated tokens per prompt")
    args = parser.parse_args()

    models = args.models or [default_word_forge_model_name()]
    model_reports = [benchmark_model(model_name, max_new_tokens=args.max_new_tokens) for model_name in models]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": model_reports,
        "recommended_model": recommend_model(model_reports),
    }

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
