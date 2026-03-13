from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "import_external_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("import_external_benchmark", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_normalize_external_benchmark_reads_summary_metrics(tmp_path: Path) -> None:
    mod = _load_module()
    source = tmp_path / "agentbench.json"
    source.write_text(
        json.dumps({"summary": {"success_rate": 0.62, "tasks_total": 100, "tasks_passed": 62}}),
        encoding="utf-8",
    )
    payload = mod.normalize_external_benchmark(
        suite="agentbench",
        input_path=source,
        source_url="https://github.com/THUDM/AgentBench",
        notes="imported",
        participant="eidos",
        execution_mode="local_run",
    )
    assert payload["suite"] == "agentbench"
    assert payload["score"] == 0.62
    assert payload["metrics"]["tasks_total"] == 100
    assert payload["participant"] == "eidos"
    assert payload["execution_mode"] == "local_run"
