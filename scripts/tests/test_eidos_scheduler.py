from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "eidos_scheduler.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader("eidos_scheduler", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader("eidos_scheduler", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


scheduler = _load_module()


def test_scheduler_once_writes_status(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(scheduler, "SCHEDULER_STATUS_PATH", runtime_dir / "scheduler.json")
    monkeypatch.setattr(
        scheduler,
        "run_pipeline",
        lambda **kwargs: {
            "run_id": "20260306_120000",
            "records_total": 5,
            "records_by_kind": {"docs": 2, "code": 3},
            "word_forge": {"enabled": True},
            "graphrag": {"indexed": True, "assessment_summary": {"status": "stable"}},
            "living_documentation": {"generated": True},
        },
    )
    monkeypatch.setattr(
        scheduler,
        "run_memory_maintenance",
        lambda repo_root, enrichment_limit, use_llm: {
            "available": True,
            "enrich_report": {"updated": 3},
            "reindex_report": {"reindexed": 4, "vector_count": 4},
            "community_summary": {"count": 2},
        },
    )
    monkeypatch.setattr(
        scheduler,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "repo_root": str(tmp_path / "repo"),
                "output_root": str(tmp_path / "output"),
                "workspace_root": str(tmp_path / "workspace"),
                "interval_sec": 30.0,
                "max_file_bytes": 1000,
                "max_chars_per_doc": 2000,
                "code_max_files": 0,
                "method": "native",
                "queries": ["status"],
                "run_graphrag": True,
                "skip_graphrag": False,
                "once": True,
                "doc_model": "qwen3.5:2b",
                "doc_thinking_mode": "on",
                "doc_timeout_sec": 900.0,
                "doc_max_tokens": 1400,
                "doc_temperature": 0.1,
                "memory_enrichment_limit": 48,
                "memory_llm_enrichment": False,
            },
        )(),
    )

    rc = scheduler.main()
    assert rc == 0
    payload = json.loads((runtime_dir / "scheduler.json").read_text(encoding="utf-8"))
    assert payload["state"] == "idle"
    assert payload["summary"]["records_total"] == 5
    assert payload["summary"]["memory"]["enrich_report"]["updated"] == 3
    assert payload["doc_model"] == "qwen3.5:2b"


def test_scheduler_degrades_model_heavy_work_when_budget_is_saturated(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(scheduler, "SCHEDULER_STATUS_PATH", runtime_dir / "scheduler.json")
    captured: dict[str, object] = {}

    def _run_pipeline(**kwargs):
        captured["run_graphrag"] = kwargs["run_graphrag"]
        cfg = kwargs["living_doc_config"]
        captured["thinking_mode"] = cfg.thinking_mode
        return {
            "run_id": "20260306_120001",
            "records_total": 2,
            "records_by_kind": {"docs": 1, "code": 1},
            "word_forge": {"enabled": True},
            "graphrag": {"indexed": False, "assessment_summary": {}},
            "living_documentation": {"generated": True},
        }

    def _run_memory(repo_root, enrichment_limit, use_llm):
        captured["memory_use_llm"] = use_llm
        return {
            "available": True,
            "enrich_report": {"updated": 1},
            "reindex_report": {"reindexed": 1, "vector_count": 1},
            "community_summary": {"count": 1},
        }

    monkeypatch.setattr(scheduler, "run_pipeline", _run_pipeline)
    monkeypatch.setattr(scheduler, "run_memory_maintenance", _run_memory)
    monkeypatch.setattr(
        scheduler,
        "_coordinator_budget",
        lambda coordinator, owner, model, memory_llm_enrichment: {
            "decision": {"allowed": False, "reason": "instance_budget_exceeded", "active_owner": "other-worker"},
            "requested_models": [{"family": "ollama", "model": model}],
            "saturated": True,
        },
    )
    monkeypatch.setattr(
        scheduler,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "repo_root": str(tmp_path / "repo"),
                "output_root": str(tmp_path / "output"),
                "workspace_root": str(tmp_path / "workspace"),
                "interval_sec": 30.0,
                "max_file_bytes": 1000,
                "max_chars_per_doc": 2000,
                "code_max_files": 0,
                "method": "native",
                "queries": ["status"],
                "run_graphrag": True,
                "skip_graphrag": False,
                "once": True,
                "doc_model": "qwen3.5:2b",
                "doc_thinking_mode": "on",
                "doc_timeout_sec": 900.0,
                "doc_max_tokens": 1400,
                "doc_temperature": 0.1,
                "memory_enrichment_limit": 48,
                "memory_llm_enrichment": True,
            },
        )(),
    )

    rc = scheduler.main()
    assert rc == 0
    assert captured["run_graphrag"] is False
    assert captured["thinking_mode"] == "off"
    assert captured["memory_use_llm"] is False
    payload = json.loads((runtime_dir / "scheduler.json").read_text(encoding="utf-8"))
    assert payload["summary"]["budget"]["saturated"] is True
