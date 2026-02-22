from __future__ import annotations

import json
from types import SimpleNamespace

from eidos_mcp.routers import consciousness


class _FakeValidator:
    def run(self, **kwargs):
        return SimpleNamespace(report={"validation_id": "val_1", "pass": True, "kwargs": kwargs})

    def latest_validation(self):
        return {"validation_id": "latest_1", "pass": True}

    def protocol_drift_review(self, **kwargs):
        return {"summary": {"flagged_count": 1}, "kwargs": kwargs}


def test_consciousness_construct_validate(monkeypatch):
    monkeypatch.setattr(consciousness, "_construct_validator", lambda state_dir=None: _FakeValidator())
    payload = json.loads(
        consciousness.consciousness_construct_validate(
            limit=7,
            persist=False,
            min_pairs=5,
            security_required=True,
        )
    )
    assert payload["validation_id"] == "val_1"
    assert payload["kwargs"]["limit"] == 7
    assert payload["kwargs"]["persist"] is False
    assert payload["kwargs"]["min_pairs"] == 5
    assert payload["kwargs"]["security_required"] is True


def test_consciousness_construct_latest(monkeypatch):
    monkeypatch.setattr(consciousness, "_construct_validator", lambda state_dir=None: _FakeValidator())
    payload = json.loads(consciousness.consciousness_construct_latest())
    assert payload["validation_id"] == "latest_1"


def test_consciousness_construct_latest_resource(monkeypatch):
    monkeypatch.setattr(consciousness, "_construct_validator", lambda state_dir=None: _FakeValidator())
    payload = json.loads(consciousness.consciousness_construct_latest_resource())
    assert payload["validation_id"] == "latest_1"


def test_consciousness_construct_drift_review(monkeypatch):
    monkeypatch.setattr(consciousness, "_construct_validator", lambda state_dir=None: _FakeValidator())
    payload = json.loads(consciousness.consciousness_construct_drift_review(threshold=0.2))
    assert payload["summary"]["flagged_count"] == 1
    assert payload["kwargs"]["threshold"] == 0.2
