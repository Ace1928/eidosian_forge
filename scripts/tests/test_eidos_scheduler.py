from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

from eidosian_runtime import ForgeRuntimeCoordinator

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "eidos_scheduler.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader("eidos_scheduler", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader("eidos_scheduler", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


mod = _load_module()


class _Proc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_scheduler_cycle_waits_on_coordinator(tmp_path: Path, monkeypatch) -> None:
    coordinator = ForgeRuntimeCoordinator(tmp_path / "forge_coordinator_status.json")
    coordinator.heartbeat(
        owner="qwenchat",
        task="interactive_chat",
        state="interactive",
        active_models=[{"family": "ollama", "model": "qwen3.5:2b", "role": "interactive_chat"}],
        metadata={"exclusive": True, "exclusive_owner": "qwenchat"},
    )
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "eidos_scheduler_status.json")
    monkeypatch.setattr(mod, "SCHEDULER_HISTORY_PATH", tmp_path / "eidos_scheduler_history.jsonl")
    monkeypatch.setattr(mod, "PROOF_STATUS_PATH", tmp_path / "entity_proof_status.json")
    monkeypatch.setattr(
        mod,
        "_refresh_directory_docs_status",
        lambda repo_root, force=False, max_age_sec=3600.0: {
            "missing_readme_count": 2,
            "coverage_ratio": 0.9,
            "missing_delta": 1,
            "coverage_delta": -0.01,
            "review_pending_count": 3,
            "suppressed_directory_count": 1,
        },
    )
    monkeypatch.setattr(
        mod, "_refresh_proof_status", lambda repo_root: {"status": "yellow", "fresh": True, "top_gap_count": 3}
    )
    result = mod.run_scheduler_cycle(
        interval_sec=30.0,
        timeout_sec=60.0,
        run_graphrag=False,
        code_max_files=None,
        repo_root=tmp_path,
        output_root=tmp_path / "out",
        workspace_root=tmp_path / "ws",
        model="qwen3.5:2b",
        coordinator=coordinator,
        cycle=1,
    )
    assert result["state"] == "waiting"
    assert result["last_result"]["reason"] == "exclusive_owner_active"
    assert result["last_result"]["proof"]["status"] == "yellow"
    history_lines = (tmp_path / "eidos_scheduler_history.jsonl").read_text(encoding="utf-8").splitlines()
    assert history_lines
    assert json.loads(history_lines[-1])["state"] == "waiting"


def test_run_scheduler_cycle_records_success(tmp_path: Path, monkeypatch) -> None:
    coordinator = ForgeRuntimeCoordinator(tmp_path / "forge_coordinator_status.json")
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "eidos_scheduler_status.json")
    monkeypatch.setattr(mod, "SCHEDULER_HISTORY_PATH", tmp_path / "eidos_scheduler_history.jsonl")
    monkeypatch.setattr(mod, "PIPELINE_STATUS_PATH", tmp_path / "living_pipeline_status.json")
    monkeypatch.setattr(mod, "DIRECTORY_DOCS_STATUS_PATH", tmp_path / "directory_docs_status.json")
    monkeypatch.setattr(mod, "DIRECTORY_DOCS_HISTORY_PATH", tmp_path / "directory_docs_history.json")
    monkeypatch.setattr(mod, "PROOF_STATUS_PATH", tmp_path / "entity_proof_status.json")
    monkeypatch.setattr(
        mod,
        "_refresh_directory_docs_status",
        lambda repo_root, force=False, max_age_sec=3600.0: {
            "missing_readme_count": 1,
            "coverage_ratio": 0.95,
            "missing_delta": -1,
            "coverage_delta": 0.02,
            "review_pending_count": 2,
            "suppressed_directory_count": 1,
        },
    )
    monkeypatch.setattr(
        mod, "_refresh_proof_status", lambda repo_root: {"status": "green", "fresh": True, "top_gap_count": 1}
    )
    (tmp_path / "living_pipeline_status.json").write_text(
        json.dumps({"phase": "indexing", "eta_seconds": 12}), encoding="utf-8"
    )

    def _fake_run(*args, **kwargs):
        return _Proc(returncode=0, stdout=json.dumps({"run_id": "run-1", "records_total": 7}), stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    result = mod.run_scheduler_cycle(
        interval_sec=30.0,
        timeout_sec=60.0,
        run_graphrag=True,
        code_max_files=5,
        repo_root=tmp_path,
        output_root=tmp_path / "out",
        workspace_root=tmp_path / "ws",
        model="qwen3.5:2b",
        coordinator=coordinator,
        cycle=2,
    )
    assert result["status"] == "success"
    assert result["run_id"] == "run-1"
    assert result["directory_docs"]["missing_readme_count"] == 1
    assert result["proof"]["status"] == "green"
    saved = json.loads((tmp_path / "eidos_scheduler_status.json").read_text(encoding="utf-8"))
    assert saved["state"] == "sleeping"
    assert saved["last_result"]["records_total"] == 7
    history_lines = (tmp_path / "eidos_scheduler_history.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["state"] == "running" for line in history_lines)
    assert json.loads(history_lines[-1])["state"] == "sleeping"


def test_scheduler_history_seeds_from_existing_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "eidos_scheduler_status.json")
    monkeypatch.setattr(mod, "SCHEDULER_HISTORY_PATH", tmp_path / "eidos_scheduler_history.jsonl")
    (tmp_path / "eidos_scheduler_status.json").write_text(
        json.dumps({"state": "sleeping", "current_task": "living_pipeline"}),
        encoding="utf-8",
    )

    mod._maybe_seed_scheduler_history()

    history_lines = (tmp_path / "eidos_scheduler_history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1
    assert json.loads(history_lines[0])["state"] == "sleeping"


def test_refresh_proof_status_reads_latest_proof_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "PROOF_STATUS_PATH", tmp_path / "entity_proof_status.json")
    latest = tmp_path / "reports" / "proof" / "entity_proof_scorecard_latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(
        json.dumps(
            {
                "overall": {"status": "yellow", "score": 0.61},
                "freshness": {"status": "yellow"},
                "regression": {"status": "stable"},
                "top_gaps": [{"gap": "a"}, {"gap": "b"}],
            }
        ),
        encoding="utf-8",
    )
    payload = mod._refresh_proof_status(tmp_path)
    assert payload["status"] == "yellow"
    assert payload["top_gap_count"] == 2


def test_refresh_directory_docs_status_tracks_drift(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "DIRECTORY_DOCS_STATUS_PATH", tmp_path / "directory_docs_status.json")
    monkeypatch.setattr(mod, "DIRECTORY_DOCS_HISTORY_PATH", tmp_path / "directory_docs_history.json")
    write_calls = [
        {
            "contract": "eidos.documentation_status.v1",
            "generated_at": "2026-03-13T00:00:00Z",
            "required_directory_count": 10,
            "missing_readme_count": 2,
            "documented_directory_count": 8,
            "coverage_ratio": 0.8,
            "missing_examples": ["a"],
        },
        {
            "contract": "eidos.documentation_status.v1",
            "generated_at": "2026-03-13T00:05:00Z",
            "required_directory_count": 10,
            "missing_readme_count": 1,
            "documented_directory_count": 9,
            "coverage_ratio": 0.9,
            "missing_examples": ["b"],
        },
    ]

    def _fake_write(repo_root, output_path):
        payload = write_calls.pop(0)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    import types

    fake_module = types.SimpleNamespace(write_inventory_status=_fake_write)
    monkeypatch.setitem(sys.modules, "doc_forge.scribe.directory_docs", fake_module)

    first = mod._refresh_directory_docs_status(tmp_path, force=True)
    second = mod._refresh_directory_docs_status(tmp_path, force=True)
    history = json.loads((tmp_path / "directory_docs_history.json").read_text(encoding="utf-8"))

    assert first["drift_state"] == "improved"
    assert second["drift_state"] == "improved"
    assert second["missing_delta"] == -1
    assert len(history["entries"]) == 2
