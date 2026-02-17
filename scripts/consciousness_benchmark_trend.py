#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def _event_ts(payload: dict[str, Any], fallback_path: Path) -> float:
    ts = payload.get("timestamp")
    if isinstance(ts, str):
        # Accept simple RFC3339 UTC timestamps like 2026-02-16T04:30:02Z.
        try:
            parsed = time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            return float(calendar.timegm(parsed))
        except ValueError:
            pass
    return fallback_path.stat().st_mtime


def _iter_reports(paths: Iterable[Path], cutoff_ts: float) -> list[tuple[float, dict[str, Any], Path]]:
    rows: list[tuple[float, dict[str, Any], Path]] = []
    for path in sorted(paths):
        payload = _load_json(path)
        if payload is None:
            continue
        event_ts = _event_ts(payload, path)
        if event_ts < cutoff_ts:
            continue
        rows.append((event_ts, payload, path))
    return sorted(rows, key=lambda item: item[0])


def _bool_rate(items: Iterable[bool]) -> float:
    values = list(items)
    if not values:
        return 0.0
    return round(sum(1 for v in values if v) / len(values), 6)


def _mean(values: Iterable[float]) -> float:
    nums = list(values)
    if not nums:
        return 0.0
    return round(float(statistics.fmean(nums)), 6)


def _latest(rows: list[tuple[float, dict[str, Any], Path]]) -> tuple[float, dict[str, Any], Path] | None:
    if not rows:
        return None
    return rows[-1]


def build_trend_report(reports_root: Path, window_days: int = 30) -> dict[str, Any]:
    now = time.time()
    window_days = max(1, int(window_days))
    cutoff_ts = now - (window_days * 86400)

    core_rows = _iter_reports((reports_root / "consciousness_benchmarks").glob("benchmark_*.json"), cutoff_ts)
    stress_rows = _iter_reports(
        (reports_root / "consciousness_stress_benchmarks").glob("stress_*.json"),
        cutoff_ts,
    )
    linux_audit_rows = _iter_reports(
        reports_root.glob("linux_audit_*.json"),
        cutoff_ts,
    )
    full_rows = _iter_reports(
        (reports_root / "consciousness_integrated_benchmarks").glob("integrated_*.json"),
        cutoff_ts,
    )
    trial_rows = _iter_reports((reports_root / "consciousness_trials").glob("trial_*.json"), cutoff_ts)
    audit_rows = _iter_reports(reports_root.glob("mcp_audit_*.json"), cutoff_ts)

    core_scores = [_safe_float((row[1].get("scores") or {}).get("composite")) for row in core_rows]
    stress_eps = [_safe_float((row[1].get("performance") or {}).get("events_per_second")) for row in stress_rows]
    stress_p95 = [_safe_float((row[1].get("performance") or {}).get("tick_latency_ms_p95")) for row in stress_rows]
    stress_trunc_rate = [
        _safe_float((row[1].get("pressure") or {}).get("truncation_rate_per_emitted_event"))
        for row in stress_rows
    ]
    linux_audit_fail_counts = [_safe_int((row[1].get("counts") or {}).get("checks_fail")) for row in linux_audit_rows]
    linux_audit_check_totals = [_safe_int((row[1].get("counts") or {}).get("checks_total")) for row in linux_audit_rows]
    full_scores = [_safe_float((row[1].get("scores") or {}).get("integrated")) for row in full_rows]
    trial_rci_delta = [_safe_float((row[1].get("delta") or {}).get("rci_delta")) for row in trial_rows]

    core_gate_rows = [(row[1].get("gates") or {}) for row in core_rows]
    stress_gate_rows = [(row[1].get("gates") or {}) for row in stress_rows]
    full_gate_rows = [(row[1].get("gates") or {}) for row in full_rows]
    core_gate_pass = _bool_rate(
        bool(g.get("world_model_online"))
        and bool(g.get("meta_online"))
        and bool(g.get("report_online"))
        and bool(g.get("latency_p95_under_100ms"))
        for g in core_gate_rows
    )
    full_gate_pass = _bool_rate(
        bool(g.get("core_score_min"))
        and bool(g.get("trial_score_min"))
        and bool(g.get("llm_success_min"))
        and bool(g.get("mcp_success_min"))
        and bool(g.get("non_regression"))
        for g in full_gate_rows
    )
    stress_gate_pass = _bool_rate(
        bool(g.get("payload_truncation_observed"))
        and bool(g.get("event_pressure_hits_target"))
        and bool(g.get("latency_p95_under_200ms"))
        and bool(g.get("module_error_free"))
        for g in stress_gate_rows
    )
    linux_audit_pass_rate = _bool_rate(count == 0 for count in linux_audit_fail_counts)

    audit_hard_fails = [
        _safe_int((row[1].get("counts") or {}).get("tool_hard_fail"))
        + _safe_int((row[1].get("counts") or {}).get("resource_hard_fail"))
        for row in audit_rows
    ]
    audit_pass_rate = _bool_rate(v == 0 for v in audit_hard_fails)

    latest_core = _latest(core_rows)
    latest_stress = _latest(stress_rows)
    latest_linux_audit = _latest(linux_audit_rows)
    latest_full = _latest(full_rows)
    latest_trial = _latest(trial_rows)
    latest_audit = _latest(audit_rows)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "window_days": window_days,
        "counts": {
            "core_benchmarks": len(core_rows),
            "stress_benchmarks": len(stress_rows),
            "linux_audits": len(linux_audit_rows),
            "integrated_benchmarks": len(full_rows),
            "trials": len(trial_rows),
            "mcp_audits": len(audit_rows),
        },
        "core_benchmark": {
            "mean_composite": _mean(core_scores),
            "latest_composite": _safe_float((latest_core[1].get("scores") or {}).get("composite")) if latest_core else None,
            "latest_id": latest_core[1].get("benchmark_id") if latest_core else None,
            "gate_pass_rate": core_gate_pass,
        },
        "stress_benchmark": {
            "mean_events_per_second": _mean(stress_eps),
            "mean_tick_latency_ms_p95": _mean(stress_p95),
            "mean_truncation_rate_per_emitted_event": _mean(stress_trunc_rate),
            "latest_id": latest_stress[1].get("benchmark_id") if latest_stress else None,
            "gate_pass_rate": stress_gate_pass,
        },
        "linux_audit": {
            "pass_rate": linux_audit_pass_rate,
            "mean_fail_count": _mean(linux_audit_fail_counts),
            "mean_checks_total": _mean(linux_audit_check_totals),
            "latest_fail_count": _safe_int((latest_linux_audit[1].get("counts") or {}).get("checks_fail")) if latest_linux_audit else None,
            "latest_checks_total": _safe_int((latest_linux_audit[1].get("counts") or {}).get("checks_total")) if latest_linux_audit else None,
            "latest_id": latest_linux_audit[1].get("run_id") if latest_linux_audit else None,
            "latest_report": str(latest_linux_audit[2]) if latest_linux_audit else None,
        },
        "integrated_benchmark": {
            "mean_integrated": _mean(full_scores),
            "latest_integrated": _safe_float((latest_full[1].get("scores") or {}).get("integrated")) if latest_full else None,
            "latest_delta": _safe_float((latest_full[1].get("scores") or {}).get("delta")) if latest_full else None,
            "latest_id": latest_full[1].get("benchmark_id") if latest_full else None,
            "gate_pass_rate": full_gate_pass,
        },
        "trials": {
            "mean_rci_delta": _mean(trial_rci_delta),
            "positive_rci_delta_rate": _bool_rate(delta >= 0.0 for delta in trial_rci_delta),
            "latest_id": latest_trial[1].get("report_id") if latest_trial else None,
        },
        "mcp_audit": {
            "hard_fail_free_rate": audit_pass_rate,
            "latest_tool_hard_fail": _safe_int((latest_audit[1].get("counts") or {}).get("tool_hard_fail")) if latest_audit else None,
            "latest_resource_hard_fail": _safe_int((latest_audit[1].get("counts") or {}).get("resource_hard_fail")) if latest_audit else None,
            "latest_report": str(latest_audit[2]) if latest_audit else None,
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    counts = report.get("counts") or {}
    core = report.get("core_benchmark") or {}
    stress = report.get("stress_benchmark") or {}
    linux_audit = report.get("linux_audit") or {}
    full = report.get("integrated_benchmark") or {}
    trials = report.get("trials") or {}
    audit = report.get("mcp_audit") or {}

    lines = [
        "# Consciousness Benchmark Trend",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Window (days): `{report.get('window_days')}`",
        "",
        "## Coverage",
        "",
        "| Series | Count |",
        "| --- | ---: |",
        f"| Core benchmark reports | {counts.get('core_benchmarks', 0)} |",
        f"| Stress benchmark reports | {counts.get('stress_benchmarks', 0)} |",
        f"| Linux audit reports | {counts.get('linux_audits', 0)} |",
        f"| Integrated benchmark reports | {counts.get('integrated_benchmarks', 0)} |",
        f"| Trial reports | {counts.get('trials', 0)} |",
        f"| MCP audit reports | {counts.get('mcp_audits', 0)} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Core mean composite | {core.get('mean_composite')} |",
        f"| Core latest composite | {core.get('latest_composite')} |",
        f"| Core gate pass rate | {core.get('gate_pass_rate')} |",
        f"| Stress mean events/s | {stress.get('mean_events_per_second')} |",
        f"| Stress mean p95 latency ms | {stress.get('mean_tick_latency_ms_p95')} |",
        f"| Stress mean truncation rate | {stress.get('mean_truncation_rate_per_emitted_event')} |",
        f"| Stress gate pass rate | {stress.get('gate_pass_rate')} |",
        f"| Linux audit pass rate | {linux_audit.get('pass_rate')} |",
        f"| Linux audit mean fail count | {linux_audit.get('mean_fail_count')} |",
        f"| Linux audit latest fail count | {linux_audit.get('latest_fail_count')} |",
        f"| Integrated mean | {full.get('mean_integrated')} |",
        f"| Integrated latest | {full.get('latest_integrated')} |",
        f"| Integrated latest delta | {full.get('latest_delta')} |",
        f"| Integrated gate pass rate | {full.get('gate_pass_rate')} |",
        f"| Trial mean RCI delta | {trials.get('mean_rci_delta')} |",
        f"| Trial positive RCI delta rate | {trials.get('positive_rci_delta_rate')} |",
        f"| MCP hard-fail-free rate | {audit.get('hard_fail_free_rate')} |",
        f"| MCP latest tool hard fails | {audit.get('latest_tool_hard_fail')} |",
        f"| MCP latest resource hard fails | {audit.get('latest_resource_hard_fail')} |",
        "",
        "## Latest IDs",
        "",
        f"- Core benchmark: `{core.get('latest_id')}`",
        f"- Stress benchmark: `{stress.get('latest_id')}`",
        f"- Linux audit: `{linux_audit.get('latest_id')}`",
        f"- Integrated benchmark: `{full.get('latest_id')}`",
        f"- Trial report: `{trials.get('latest_id')}`",
        f"- MCP audit report: `{audit.get('latest_report')}`",
        "",
    ]
    return "\n".join(lines)


def _default_output_dir(reports_root: Path) -> Path:
    out_dir = reports_root / "consciousness_trends"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate aggregate trend reports for consciousness benchmarks.")
    parser.add_argument(
        "--reports-root",
        default="reports",
        help="Reports root directory (default: reports).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Window in days to include in trend metrics (default: 30).",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path. Defaults to reports/consciousness_trends/latest_trend.json",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional output Markdown path. Defaults to reports/consciousness_trends/latest_trend.md",
    )
    parser.add_argument("--stdout-json", action="store_true", help="Also print JSON report to stdout.")
    args = parser.parse_args()

    reports_root = Path(args.reports_root).resolve()
    trend = build_trend_report(reports_root=reports_root, window_days=max(1, args.window_days))

    out_dir = _default_output_dir(reports_root)
    out_json = Path(args.output_json).resolve() if args.output_json else (out_dir / "latest_trend.json")
    out_md = Path(args.output_md).resolve() if args.output_md else (out_dir / "latest_trend.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(trend, indent=2, default=str), encoding="utf-8")
    out_md.write_text(render_markdown(trend), encoding="utf-8")

    print(f"Trend JSON: {out_json}")
    print(f"Trend Markdown: {out_md}")
    if args.stdout_json:
        print(json.dumps(trend, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
