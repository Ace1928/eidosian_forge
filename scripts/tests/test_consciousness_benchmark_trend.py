from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "consciousness_benchmark_trend.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader("consciousness_benchmark_trend", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader("consciousness_benchmark_trend", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


trend = _load_module()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_trend_report_aggregates_expected_fields(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_json(
        reports / "consciousness_benchmarks" / "benchmark_test.json",
        {
            "timestamp": "2026-02-16T00:00:00Z",
            "benchmark_id": "benchmark_test",
            "scores": {"composite": 0.71},
            "gates": {
                "world_model_online": True,
                "meta_online": True,
                "report_online": True,
                "latency_p95_under_100ms": True,
            },
        },
    )
    _write_json(
        reports / "consciousness_integrated_benchmarks" / "integrated_test.json",
        {
            "timestamp": "2026-02-16T00:01:00Z",
            "benchmark_id": "integrated_test",
            "scores": {"integrated": 0.68, "delta": 0.04},
            "gates": {
                "core_score_min": True,
                "trial_score_min": True,
                "llm_success_min": True,
                "mcp_success_min": True,
                "non_regression": True,
            },
        },
    )
    _write_json(
        reports / "consciousness_stress_benchmarks" / "stress_test.json",
        {
            "timestamp": "2026-02-16T00:01:30Z",
            "benchmark_id": "stress_test",
            "performance": {"events_per_second": 120.0, "tick_latency_ms_p95": 45.0},
            "pressure": {"truncation_rate_per_emitted_event": 0.11},
            "gates": {
                "payload_truncation_observed": True,
                "event_pressure_hits_target": True,
                "latency_p95_under_200ms": True,
                "module_error_free": True,
            },
        },
    )
    _write_json(
        reports / "consciousness_trials" / "trial_test.json",
        {
            "timestamp": "2026-02-16T00:02:00Z",
            "report_id": "trial_test",
            "delta": {"rci_delta": 0.11},
        },
    )
    _write_json(
        reports / "mcp_audit_20260216_000000.json",
        {
            "timestamp": "2026-02-16T00:03:00Z",
            "counts": {"tool_hard_fail": 0, "resource_hard_fail": 0},
        },
    )
    _write_json(
        reports / "linux_audit_20260216_000100.json",
        {
            "timestamp": "2026-02-16T00:04:00Z",
            "run_id": "linux_audit_test",
            "counts": {"checks_total": 5, "checks_fail": 0},
        },
    )

    result = trend.build_trend_report(reports_root=reports, window_days=3650)
    assert result["counts"]["core_benchmarks"] == 1
    assert result["counts"]["stress_benchmarks"] == 1
    assert result["counts"]["linux_audits"] == 1
    assert result["counts"]["integrated_benchmarks"] == 1
    assert result["counts"]["trials"] == 1
    assert result["counts"]["mcp_audits"] == 1
    assert result["core_benchmark"]["latest_id"] == "benchmark_test"
    assert result["stress_benchmark"]["latest_id"] == "stress_test"
    assert result["linux_audit"]["latest_id"] == "linux_audit_test"
    assert result["linux_audit"]["pass_rate"] == 1.0
    assert result["integrated_benchmark"]["latest_id"] == "integrated_test"
    assert result["trials"]["latest_id"] == "trial_test"
    assert result["mcp_audit"]["hard_fail_free_rate"] == 1.0


def test_main_writes_json_and_markdown_outputs(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_json(
        reports / "consciousness_benchmarks" / "benchmark_test.json",
        {
            "timestamp": "2026-02-16T00:00:00Z",
            "benchmark_id": "benchmark_test",
            "scores": {"composite": 0.55},
            "gates": {
                "world_model_online": True,
                "meta_online": False,
                "report_online": True,
                "latency_p95_under_100ms": True,
            },
        },
    )
    _write_json(
        reports / "consciousness_stress_benchmarks" / "stress_test.json",
        {
            "timestamp": "2026-02-16T00:00:30Z",
            "benchmark_id": "stress_test",
            "performance": {"events_per_second": 90.0, "tick_latency_ms_p95": 55.0},
            "pressure": {"truncation_rate_per_emitted_event": 0.14},
            "gates": {
                "payload_truncation_observed": True,
                "event_pressure_hits_target": True,
                "latency_p95_under_200ms": True,
                "module_error_free": True,
            },
        },
    )
    _write_json(
        reports / "linux_audit_20260216_000100.json",
        {
            "timestamp": "2026-02-16T00:01:00Z",
            "run_id": "linux_audit_test",
            "counts": {"checks_total": 5, "checks_fail": 0},
        },
    )

    out_json = reports / "consciousness_trends" / "custom.json"
    out_md = reports / "consciousness_trends" / "custom.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--reports-root",
            str(reports),
            "--window-days",
            "3650",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["counts"]["core_benchmarks"] == 1
    assert payload["counts"]["stress_benchmarks"] == 1
    assert payload["counts"]["linux_audits"] == 1
    md = out_md.read_text(encoding="utf-8")
    assert "# Consciousness Benchmark Trend" in md
    assert "Stress mean events/s" in md
    assert "Linux audit pass rate" in md
