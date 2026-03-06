from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "run_graphrag_bench.py"
spec = importlib.util.spec_from_file_location("run_graphrag_bench", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_validate_pipeline_outputs_rejects_placeholders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "WORKSPACE_DIR", tmp_path / "workspace")
    reports_path = mod.WORKSPACE_DIR / "output" / "native_community_reports.json"
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    reports_path.write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "title": "Community 0",
                        "summary": "Auto-generated placeholder summary for Community 0.",
                        "findings": ["Fallback generated for missing data."],
                        "rating": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Placeholder/fallback community reports detected"):
        mod.validate_pipeline_outputs()


def test_validate_pipeline_outputs_accepts_grounded_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "WORKSPACE_DIR", tmp_path / "workspace")
    reports_path = mod.WORKSPACE_DIR / "output" / "native_community_reports.json"
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    reports_path.write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "title": "Alaric and Seraphina in Eidos",
                        "summary": "Alaric and Seraphina coordinate defense of the Crystal in Eidos.",
                        "findings": ["Kael protects the Crystal while Seraphina advises Alaric in Eidos."],
                        "rating": 8.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = mod.validate_pipeline_outputs()
    assert result["report_source"] == "native_json"
    assert result["placeholder_rows"] == 0
    assert result["relevance_fail_rows"] == 0
    assert result["schema_fail_rows"] == 0


def test_run_benchmark_rejects_placeholder_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIDOS_GRAPHRAG_ALLOW_PLACEHOLDER", "1")
    with pytest.raises(RuntimeError, match="must be disabled"):
        mod.run_benchmark("test query")
