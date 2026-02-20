from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "linux_audit_matrix.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader(
        "linux_audit_matrix",
        str(SCRIPT_PATH),
    )
    spec = importlib.util.spec_from_loader("linux_audit_matrix", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


audit = _load_module()


def test_extract_report_path_parses_expected_line() -> None:
    text = "Report: /tmp/example/report.json\nok"
    out = audit._extract_report_path(text)
    assert out is not None
    assert out.name == "report.json"


def test_evaluate_forge_status_flags_unavailable_forge() -> None:
    ok, details = audit.evaluate_forge_status(
        {
            "forges": {
                "agent_forge": {"available": True, "status": "ok"},
                "memory_forge": {"available": False, "status": "error"},
            }
        }
    )
    assert ok is False
    assert "memory_forge" in details


def test_evaluate_mcp_audit_strict_rejects_soft_fail() -> None:
    payload = {
        "counts": {
            "tool_hard_fail": 0,
            "resource_hard_fail": 0,
            "tool_soft_fail": 1,
        }
    }
    ok, _ = audit.evaluate_mcp_audit(payload, strict=False)
    assert ok is True
    ok_strict, details_strict = audit.evaluate_mcp_audit(payload, strict=True)
    assert ok_strict is False
    assert "soft fails" in details_strict


def test_evaluate_consciousness_status_requires_watchdog_and_payload_safety() -> None:
    missing = {"workspace": {}, "coherence": {}, "rci": {}}
    ok, details = audit.evaluate_consciousness_status(missing)
    assert ok is False
    assert "watchdog" in details


def test_evaluate_stress_benchmark_checks_truncation_and_gates() -> None:
    payload = {
        "pressure": {"payload_truncations_observed": 3},
        "gates": {
            "event_pressure_hits_target": True,
            "latency_p95_under_200ms": True,
            "module_error_free": True,
        },
    }
    ok, details = audit.evaluate_stress_benchmark(payload)
    assert ok is True
    assert "truncations=3" in details


def test_evaluate_stress_benchmark_strict_latency_gate() -> None:
    payload = {
        "pressure": {"payload_truncations_observed": 2},
        "gates": {
            "event_pressure_hits_target": True,
            "latency_p95_under_200ms": False,
            "module_error_free": True,
        },
    }
    ok_relaxed, details_relaxed = audit.evaluate_stress_benchmark(payload)
    assert ok_relaxed is True
    assert "warn" in details_relaxed

    ok_strict, details_strict = audit.evaluate_stress_benchmark(payload, strict_latency=True)
    assert ok_strict is False
    assert "latency" in details_strict
