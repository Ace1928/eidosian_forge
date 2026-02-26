from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "graphrag_model_sweep.py"
spec = importlib.util.spec_from_file_location("graphrag_model_sweep", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_score_tuple_prefers_successful_higher_score() -> None:
    failed = mod.SweepResult(
        model_id="failed",
        model_path="x",
        bench_ok=False,
        bench_returncode=1,
        bench_stdout_tail="",
        metrics_path=None,
        assessment_path=None,
        final_score=0.9,
        rank="A",
        index_seconds=10.0,
        query_seconds=10.0,
        query_output="",
    )
    successful = mod.SweepResult(
        model_id="ok",
        model_path="y",
        bench_ok=True,
        bench_returncode=0,
        bench_stdout_tail="",
        metrics_path=None,
        assessment_path=None,
        final_score=0.4,
        rank="C",
        index_seconds=10.0,
        query_seconds=10.0,
        query_output="answer",
    )
    assert mod._score_tuple(successful) > mod._score_tuple(failed)


def test_score_tuple_uses_runtime_as_tiebreaker() -> None:
    slower = mod.SweepResult(
        model_id="slow",
        model_path="x",
        bench_ok=True,
        bench_returncode=0,
        bench_stdout_tail="",
        metrics_path=None,
        assessment_path=None,
        final_score=0.8,
        rank="B",
        index_seconds=50.0,
        query_seconds=10.0,
        query_output="answer",
    )
    faster = mod.SweepResult(
        model_id="fast",
        model_path="y",
        bench_ok=True,
        bench_returncode=0,
        bench_stdout_tail="",
        metrics_path=None,
        assessment_path=None,
        final_score=0.8,
        rank="B",
        index_seconds=20.0,
        query_seconds=5.0,
        query_output="answer",
    )
    assert mod._score_tuple(faster) > mod._score_tuple(slower)


def test_discover_models_filters_non_llm_files(tmp_path, monkeypatch) -> None:
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "Qwen2.5-0.5B-Instruct-Q8_0.gguf").write_text("", encoding="utf-8")
    (models / "my-embed-model.gguf").write_text("", encoding="utf-8")
    (models / "vision-mmproj.gguf").write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "MODELS_DIR", models)
    discovered = mod.discover_models()
    assert len(discovered) == 1
    assert discovered[0][0].startswith("qwen2_5_0_5b_instruct")


def test_main_rejects_invalid_model_arg(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["graphrag_model_sweep.py", "--model", "badspec"])
    with pytest.raises(SystemExit):
        mod.main()
