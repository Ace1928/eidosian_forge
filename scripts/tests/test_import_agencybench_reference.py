from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "import_agencybench_reference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("import_agencybench_reference", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_aggregate_agencybench_samples(tmp_path: Path) -> None:
    mod = _load_module()
    sample = tmp_path / "AgencyBench-v2" / "MCP" / "scenario1" / "claude"
    sample.mkdir(parents=True, exist_ok=True)
    (sample / "meta_eval.json").write_text(
        json.dumps(
            {
                "subtasks": [
                    {"success": True},
                    {"success": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = mod.aggregate_agencybench(tmp_path)
    assert payload["suite"] == "agencybench"
    assert payload["execution_mode"] == "imported_reference"
    assert payload["metrics"]["tasks_total"] == 1
    assert payload["metrics"]["subtasks_total"] == 2
    assert payload["score"] == 0.5
