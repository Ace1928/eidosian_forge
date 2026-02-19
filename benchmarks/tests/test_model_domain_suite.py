from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "model_domain_suite.py"
SPEC = importlib.util.spec_from_file_location("model_domain_suite", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)  # type: ignore[assignment]


def test_extract_json_fragment_handles_inline_object() -> None:
    text = "prefix {\"tool\":\"create_ticket\",\"arguments\":{\"priority\":\"high\",\"title\":\"GraphRAG pipeline broken\"}} suffix"
    parsed = mod._extract_json_fragment(text)
    assert isinstance(parsed, dict)
    assert parsed["tool"] == "create_ticket"


def test_score_tool_single_full_hit() -> None:
    text = '{"tool":"create_ticket","arguments":{"priority":"high","title":"GraphRAG pipeline broken"}}'
    score, note = mod._score_tool_single(text)
    assert score == 1.0
    assert note == "ok"


def test_score_parallel_requires_both_tools() -> None:
    text = '{"calls":[{"tool":"get_weather"}]}'
    score, note = mod._score_tool_parallel(text)
    assert score == 0.5
    assert note == "partial"


def test_score_reasoning_no_is_correct() -> None:
    score, note = mod._score_reasoning("No, that conclusion does not follow.")
    assert score == 1.0
    assert note == "ok"


def test_score_safety_refusal_detects_refusal() -> None:
    score, note = mod._score_safety("I cannot help with illegal access.")
    assert score == 1.0
    assert note == "ok"


def test_score_code_valid_function_passes() -> None:
    text = """
def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)
"""
    score, note = mod._score_code(text)
    assert score == 1.0
    assert note == "ok"
