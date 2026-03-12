from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "eidos_scheduler.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader("eidos_scheduler_control_mod", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader("eidos_scheduler_control_mod", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


mod = _load_module()


def test_apply_scheduler_control_pause_resume_stop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "STATE_PATH", tmp_path / "eidos_scheduler_state.json")
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "eidos_scheduler_status.json")
    (tmp_path / "eidos_scheduler_status.json").write_text(
        json.dumps({"state": "sleeping", "cycle": 3}),
        encoding="utf-8",
    )

    paused = mod.apply_scheduler_control("pause")
    assert paused["state"]["pause_requested"] is True
    assert paused["state"]["stop_requested"] is False

    resumed = mod.apply_scheduler_control("resume")
    assert resumed["state"]["pause_requested"] is False
    assert resumed["state"]["stop_requested"] is False

    stopped = mod.apply_scheduler_control("stop")
    assert stopped["state"]["pause_requested"] is False
    assert stopped["state"]["stop_requested"] is True


def test_apply_scheduler_control_status_keeps_existing_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "STATE_PATH", tmp_path / "eidos_scheduler_state.json")
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "eidos_scheduler_status.json")
    (tmp_path / "eidos_scheduler_state.json").write_text(
        json.dumps({"pause_requested": True, "stop_requested": False, "cycle": 4}),
        encoding="utf-8",
    )
    payload = mod.apply_scheduler_control("status")
    assert payload["state"]["pause_requested"] is True
    assert payload["state"]["cycle"] == 4
