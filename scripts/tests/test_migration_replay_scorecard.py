from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "migration_replay_scorecard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("migration_replay_scorecard", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_scorecard_scores_boot_and_replay_contracts(tmp_path: Path) -> None:
    mod = _load_module()
    _write_json(tmp_path / "data" / "runtime" / "platform_capabilities.json", {"platform": "termux"})
    _write_json(tmp_path / "data" / "runtime" / "eidos_scheduler_status.json", {"state": "sleeping"})
    _write_json(tmp_path / "data" / "runtime" / "forge_coordinator_status.json", {"state": "idle"})
    _write_json(tmp_path / "reports" / "proof" / "entity_proof_scorecard_latest.json", {"overall": {"score": 0.7}})
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "THEORY_OF_OPERATION.md").write_text("# Theory\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "install_termux_runit_services.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "scripts" / "install_termux_boot.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "scripts" / "eidos_scheduler_control.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    payload = mod.build_scorecard(tmp_path)

    assert payload["contract"] == "eidos.migration_replay_scorecard.v1"
    assert payload["overall_score"] > 0.5
    assert payload["checks"]["platform_capabilities"] is True
