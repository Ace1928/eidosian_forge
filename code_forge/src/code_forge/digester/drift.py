from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _flatten_numeric(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_numeric(value, path))
            continue
        num = _safe_float(value)
        if num is None:
            continue
        out[path] = num
    return out


def build_metrics_snapshot(
    *,
    ingestion_stats: dict[str, Any],
    triage: dict[str, Any],
    duplication: dict[str, Any],
    dependency_graph: dict[str, Any],
    relationship_counts: dict[str, int],
) -> dict[str, Any]:
    entries = triage.get("entries") or []
    confidence_vals: list[float] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        val = _safe_float(entry.get("confidence"))
        if val is not None:
            confidence_vals.append(val)

    avg_conf = (sum(confidence_vals) / len(confidence_vals)) if confidence_vals else 0.0
    return {
        "generated_at": _utc_now(),
        "ingestion": {
            "files_processed": int(ingestion_stats.get("files_processed") or 0),
            "units_created": int(ingestion_stats.get("units_created") or 0),
            "errors": int(ingestion_stats.get("errors") or 0),
            "elapsed_seconds": float(ingestion_stats.get("elapsed_seconds") or 0.0),
            "files_skipped": int(ingestion_stats.get("files_skipped") or 0),
        },
        "triage": {
            "label_counts": triage.get("label_counts") or {},
            "entry_count": len(entries),
            "avg_confidence": round(avg_conf, 6),
        },
        "duplication": {
            "summary": duplication.get("summary") or {},
        },
        "dependency_graph": {
            "summary": dependency_graph.get("summary") or {},
        },
        "relationships": relationship_counts or {},
    }


def _compare_metrics(
    current: dict[str, Any],
    previous: dict[str, Any],
    *,
    warn_pct: float = 30.0,
    min_abs_delta: float = 1.0,
) -> dict[str, Any]:
    cur_flat = _flatten_numeric(current)
    prev_flat = _flatten_numeric(previous)

    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []

    for key in sorted(set(cur_flat).intersection(prev_flat)):
        cur = cur_flat[key]
        prev = prev_flat[key]
        delta = cur - prev
        pct = 0.0
        if prev != 0:
            pct = (delta / prev) * 100.0

        comparisons.append(
            {
                "metric": key,
                "current": round(cur, 6),
                "previous": round(prev, 6),
                "delta": round(delta, 6),
                "delta_pct": round(pct, 4),
            }
        )

        if abs(delta) >= min_abs_delta and abs(pct) >= warn_pct:
            direction = "increased" if delta > 0 else "decreased"
            warnings.append(
                f"{key} {direction} by {pct:.2f}% (current={cur:.4f}, previous={prev:.4f})"
            )

    return {
        "compared_metric_count": len(comparisons),
        "warn_pct": warn_pct,
        "min_abs_delta": min_abs_delta,
        "comparisons": comparisons,
        "warnings": warnings,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Code Forge Drift Report",
        "",
        f"Generated: {report.get('generated_at')}",
        f"Output directory: `{report.get('output_dir')}`",
        f"Current snapshot: `{report.get('current_snapshot_path')}`",
        f"Previous snapshot: `{report.get('previous_snapshot_path')}`",
        "",
        f"Compared metrics: {report.get('comparison', {}).get('compared_metric_count', 0)}",
        "",
        "## Warnings",
        "",
    ]

    warnings = report.get("comparison", {}).get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Top Changes")
    lines.append("")

    changes = report.get("comparison", {}).get("comparisons", [])
    sorted_changes = sorted(changes, key=lambda row: abs(float(row.get("delta_pct") or 0.0)), reverse=True)[:20]
    for row in sorted_changes:
        lines.append(
            f"- `{row.get('metric')}`: {row.get('previous')} -> {row.get('current')} (delta={row.get('delta')}, pct={row.get('delta_pct')}%)"
        )

    return "\n".join(lines) + "\n"


def load_latest_history_snapshot(history_dir: Path) -> Optional[dict[str, Any]]:
    history_dir = Path(history_dir)
    if not history_dir.exists():
        return None
    files = sorted(history_dir.glob("*.json"))
    if not files:
        return None
    latest = files[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload["_path"] = str(latest)
        return payload
    return None


def write_history_snapshot(
    *,
    history_dir: Path,
    run_id: str,
    metrics: dict[str, Any],
    summary_path: Path,
) -> Path:
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_run_id = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in str(run_id))
    safe_run_id = (safe_run_id or "run")[:96]
    target = history_dir / f"{ts}_{safe_run_id}.json"
    payload = {
        "generated_at": _utc_now(),
        "run_id": run_id,
        "summary_path": str(summary_path),
        "metrics": metrics,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def write_drift_report(
    *,
    output_dir: Path,
    current_snapshot_path: Path,
    current_metrics: dict[str, Any],
    previous_snapshot: Optional[dict[str, Any]],
    warn_pct: float = 30.0,
    min_abs_delta: float = 1.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_metrics = {}
    previous_path: Optional[str] = None
    if previous_snapshot:
        previous_metrics = previous_snapshot.get("metrics") or {}
        previous_path = previous_snapshot.get("_path")

    comparison = _compare_metrics(
        current=current_metrics,
        previous=previous_metrics,
        warn_pct=float(warn_pct),
        min_abs_delta=float(min_abs_delta),
    ) if previous_metrics else {
        "compared_metric_count": 0,
        "warn_pct": float(warn_pct),
        "min_abs_delta": float(min_abs_delta),
        "comparisons": [],
        "warnings": [],
    }

    report = {
        "generated_at": _utc_now(),
        "output_dir": str(Path(output_dir).resolve()),
        "current_snapshot_path": str(current_snapshot_path),
        "previous_snapshot_path": previous_path,
        "comparison": comparison,
    }

    json_path = output_dir / "drift_report.json"
    md_path = output_dir / "drift_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    report["drift_report_json_path"] = str(json_path)
    report["drift_report_md_path"] = str(md_path)
    return report


def build_drift_report_from_output(
    *,
    output_dir: Path,
    previous_snapshot_path: Optional[Path] = None,
    history_dir: Optional[Path] = None,
    write_history: bool = True,
    run_id: Optional[str] = None,
    warn_pct: float = 30.0,
    min_abs_delta: float = 1.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "archive_digester_summary.json"
    triage_path = output_dir / "triage.json"
    duplication_path = output_dir / "duplication_index.json"
    dep_graph_path = output_dir / "dependency_graph.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    triage = json.loads(triage_path.read_text(encoding="utf-8")) if triage_path.exists() else {}
    duplication = json.loads(duplication_path.read_text(encoding="utf-8")) if duplication_path.exists() else {}
    dependency = json.loads(dep_graph_path.read_text(encoding="utf-8")) if dep_graph_path.exists() else {}

    current_metrics = build_metrics_snapshot(
        ingestion_stats=summary.get("ingestion_stats") or {},
        triage=triage,
        duplication=duplication,
        dependency_graph=dependency,
        relationship_counts=summary.get("relationship_counts") or {},
    )

    history_target = Path(history_dir) if history_dir is not None else (output_dir / "history")

    previous_snapshot = None
    if previous_snapshot_path is not None:
        data = json.loads(Path(previous_snapshot_path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data["_path"] = str(Path(previous_snapshot_path).resolve())
            previous_snapshot = data
    else:
        history_latest = load_latest_history_snapshot(history_target)
        if history_latest is not None:
            previous_snapshot = history_latest

    report = write_drift_report(
        output_dir=output_dir,
        current_snapshot_path=summary_path,
        current_metrics=current_metrics,
        previous_snapshot=previous_snapshot,
        warn_pct=float(warn_pct),
        min_abs_delta=float(min_abs_delta),
    )

    if write_history:
        snapshot_run_id = str(run_id or summary.get("generated_at") or "run")
        history_snapshot_path = write_history_snapshot(
            history_dir=history_target,
            run_id=snapshot_run_id,
            metrics=current_metrics,
            summary_path=summary_path,
        )
        report["history_snapshot_path"] = str(history_snapshot_path)

    return report
