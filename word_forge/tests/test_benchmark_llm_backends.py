from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_llm_backends.py"
SPEC = importlib.util.spec_from_file_location("wf_benchmark_llm_backends", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PromptCase = MODULE.PromptCase
benchmark_model = MODULE.benchmark_model
extract_json_object = MODULE.extract_json_object
score_case = MODULE.score_case


def test_extract_json_object_prefers_last_valid_object() -> None:
    text = 'schema {"bad": [string]} output {"phrases": ["atlas"], "entities": ["eidos"]}'
    parsed = extract_json_object(text)
    assert parsed == {"phrases": ["atlas"], "entities": ["eidos"]}


def test_score_case_requires_expected_keys() -> None:
    case = PromptCase(name="term_extraction", prompt="x", expects_json=True, required_keys=("phrases", "entities"))
    result = score_case(case, '{"phrases": ["atlas"]}', latency_s=2.0)
    assert result["valid"] is False
    assert result["score"] == 0.0


def test_benchmark_model_uses_supplied_model_state(monkeypatch) -> None:
    class FakeState:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def initialize(self) -> bool:
            return True

        def generate_text(self, prompt: str, max_new_tokens: int = 96, temperature: float = 0.2):
            if "phrases" in prompt:
                return '{"phrases": ["atlas"], "entities": ["eidos"]}'
            if "ipa" in prompt:
                return '{"ipa": "/x/", "arpabet": "X", "stress_pattern": "1"}'
            return "Serendipity means a fortunate accidental discovery."

    monkeypatch.setattr(MODULE, "ModelState", FakeState)
    report = benchmark_model("gguf:/tmp/demo.gguf")
    assert report["initialized"] is True
    assert report["score"] == 1.0
    assert len(report["cases"]) == 3
