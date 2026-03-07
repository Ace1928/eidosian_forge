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


def test_run_scheduler_cycle_records_success(tmp_path: Path, monkeypatch) -> None:
    coordinator = ForgeRuntimeCoordinator(tmp_path / "forge_coordinator_status.json")
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "eidos_scheduler_status.json")
    monkeypatch.setattr(mod, "PIPELINE_STATUS_PATH", tmp_path / "living_pipeline_status.json")
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
    saved = json.loads((tmp_path / "eidos_scheduler_status.json").read_text(encoding="utf-8"))
    assert saved["state"] == "sleeping"
    assert saved["last_result"]["records_total"] == 7
