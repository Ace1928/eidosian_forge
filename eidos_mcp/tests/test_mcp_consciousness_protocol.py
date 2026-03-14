from __future__ import annotations

from pathlib import Path

import pytest
from eidos_mcp.consciousness_protocol import ConsciousnessProtocol


def _probe_ok() -> tuple[bool, dict]:
    return True, {"probe": "ok"}


def _probe_fail() -> tuple[bool, dict]:
    return False, {"probe": "fail"}


def test_bootstrap_hypotheses_file(tmp_path: Path) -> None:
    protocol = ConsciousnessProtocol(
        root_dir=tmp_path,
        probe_registry={"ok_probe": _probe_ok},
        min_tool_count=1,
        min_resource_count=1,
    )

    hypothesis_path = tmp_path / "data" / "consciousness" / "hypotheses.json"
    assert hypothesis_path.exists()
    hypotheses = protocol.list_hypotheses(active_only=False)
    assert len(hypotheses) >= 1


def test_upsert_hypothesis_and_assessment(tmp_path: Path) -> None:
    protocol = ConsciousnessProtocol(
        root_dir=tmp_path,
        probe_registry={"ok_probe": _probe_ok, "fail_probe": _probe_fail},
        min_tool_count=1,
        min_resource_count=1,
    )

    hypothesis = protocol.upsert_hypothesis(
        name="Stub Pass Threshold",
        statement="Stub mix should pass a >= 0.5 threshold.",
        metric="probe_success_rate",
        comparator=">=",
        threshold=0.5,
        source_url="https://example.com/hypothesis",
    )

    report = protocol.run_assessment(trials=2, persist=True)
    assert report["report_id"]
    assert report["metrics"]["probe_success_rate"] == pytest.approx(0.5)

    result = next(item for item in report["hypothesis_results"] if item["id"] == hypothesis["id"])
    assert result["status"] == "supported"
    assert result["passed"] is True

    loaded = protocol.get_report(report["report_id"])
    assert loaded is not None
    assert loaded["report_id"] == report["report_id"]


def test_latest_report_roundtrip(tmp_path: Path) -> None:
    protocol = ConsciousnessProtocol(
        root_dir=tmp_path,
        probe_registry={"ok_probe": _probe_ok},
        min_tool_count=1,
        min_resource_count=1,
    )

    first = protocol.run_assessment(trials=1, persist=True)
    second = protocol.run_assessment(trials=1, persist=True)
    latest = protocol.latest_report()

    assert latest is not None
    assert latest["report_id"] in {first["report_id"], second["report_id"]}
