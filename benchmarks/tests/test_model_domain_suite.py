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


def test_score_ambiguity_resolution_requires_missing_details() -> None:
    text = '{"intent":"book a restaurant table","ambiguity":"Missing city and party size."}'
    score, note = mod._score_ambiguity_resolution(text)
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


def test_expand_ollama_specs_materializes_thinking_modes() -> None:
    specs = mod._expand_ollama_specs(["qwen=qwen3.5:2b"], ["off", "on"])
    assert [spec.model_id for spec in specs] == ["qwen@off", "qwen@on"]
    assert all(spec.provider == "ollama" for spec in specs)
    assert [spec.thinking_mode for spec in specs] == ["off", "on"]


def test_normalize_ollama_chat_response_extracts_thinking_channel() -> None:
    content, thinking = mod._normalize_ollama_chat_response(
        {"message": {"role": "assistant", "content": "READY", "thinking": "checked state"}}
    )
    assert content == "READY"
    assert thinking == "checked state"
