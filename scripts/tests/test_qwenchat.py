from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

from eidosian_runtime import ForgeRuntimeCoordinator

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "qwenchat.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader("qwenchat", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader("qwenchat", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


mod = _load_module()


def test_qwenchat_wait_message_uses_eta(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "PIPELINE_STATUS_PATH", runtime_dir / "living_pipeline_status.json")
    monkeypatch.setattr(mod, "SCHEDULER_STATUS_PATH", runtime_dir / "eidos_scheduler_status.json")
    (runtime_dir / "living_pipeline_status.json").write_text(
        json.dumps({"phase": "doc_forge", "eta_seconds": 42}),
        encoding="utf-8",
    )
    message = mod._wait_message(
        {"active_owner": "eidos_scheduler"}, {"owner": "eidos_scheduler", "task": "living_pipeline"}
    )
    assert "doc_forge" in message
    assert "42s" in message


def test_exclusive_qwenchat_owner_blocks_other_allocations(tmp_path: Path) -> None:
    coordinator = ForgeRuntimeCoordinator(tmp_path / "forge_coordinator_status.json")
    coordinator.heartbeat(
        owner="qwenchat",
        task="interactive_chat",
        state="interactive",
        active_models=[{"family": "ollama", "model": "qwen3.5:2b", "role": "interactive_chat"}],
        metadata={"exclusive": True, "exclusive_owner": "qwenchat"},
    )
    decision = coordinator.can_allocate(
        owner="eidos_scheduler",
        requested_models=[{"family": "ollama", "model": "qwen3.5:2b", "role": "living_documentation"}],
        allow_same_owner=False,
    )
    assert decision["allowed"] is False
    assert decision["reason"] == "exclusive_owner_active"


def test_coordinator_recovers_dead_pid_owner(tmp_path: Path) -> None:
    coordinator = ForgeRuntimeCoordinator(tmp_path / "forge_coordinator_status.json")
    coordinator.heartbeat(
        owner="local_mcp_agent:agencybench_scenario2",
        task="local_agent:agencybench_scenario2",
        state="running",
        active_models=[{"family": "ollama", "model": "qwen3.5:2b", "role": "local_agent:agencybench_scenario2"}],
        metadata={"mode": "local_agent_cycle", "pid": 99999999},
    )
    recovery = coordinator.recover_stale_owner(reason="test_recovery")
    assert recovery["released"] is True
    payload = coordinator.read()
    assert payload["owner"] == ""
    assert payload["state"] == "idle"
