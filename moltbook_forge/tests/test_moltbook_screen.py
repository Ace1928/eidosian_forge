from __future__ import annotations

import json
from pathlib import Path

from moltbook_forge.moltbook_screen import main


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_screen_quarantines_on_risk(tmp_path: Path) -> None:
    payload = {"risk_score": 0.9, "flags": []}
    path = _write_payload(tmp_path, payload)
    output = tmp_path / "decision.json"
    assert main(["--input", str(path), "--output", str(output), "--threshold", "0.4"]) == 0
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["decision"] == "quarantine"
    assert decision["reason"] == "risk_threshold"


def test_screen_quarantines_on_critical_flag(tmp_path: Path) -> None:
    payload = {"risk_score": 0.1, "flags": ["system prompt"]}
    path = _write_payload(tmp_path, payload)
    output = tmp_path / "decision.json"
    assert main(["--input", str(path), "--output", str(output), "--threshold", "0.8"]) == 0
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["decision"] == "quarantine"
    assert decision["reason"] == "critical_flag"


def test_screen_allows_below_threshold(tmp_path: Path) -> None:
    payload = {"risk_score": 0.1, "flags": []}
    path = _write_payload(tmp_path, payload)
    output = tmp_path / "decision.json"
    assert main(["--input", str(path), "--output", str(output), "--threshold", "0.4"]) == 0
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["decision"] == "allow"
