from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

SCHEMA_VERSION = "code_forge_provenance_registry_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _build_unit_link_index(provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
    knowledge_links = _safe_list(_safe_dict(provenance.get("knowledge_links")).get("links"))
    memory_links = _safe_list(_safe_dict(provenance.get("memory_links")).get("links"))
    graphrag_docs = _safe_list(_safe_dict(provenance.get("graphrag_links")).get("documents"))

    by_unit: dict[str, dict[str, Any]] = {}

    def _entry(unit_id: str) -> dict[str, Any]:
        rec = by_unit.get(unit_id)
        if rec is None:
            rec = {
                "unit_id": unit_id,
                "knowledge_node_id": None,
                "knowledge_status": None,
                "memory_id": None,
                "memory_status": None,
                "graphrag_document_path": None,
                "qualified_name": None,
                "language": None,
                "unit_type": None,
                "source_file_path": None,
            }
            by_unit[unit_id] = rec
        return rec

    for link in knowledge_links:
        if not isinstance(link, Mapping):
            continue
        unit_id = str(link.get("unit_id") or "").strip()
        if not unit_id:
            continue
        rec = _entry(unit_id)
        rec["knowledge_node_id"] = str(link.get("node_id") or "") or None
        rec["knowledge_status"] = str(link.get("status") or "") or None

    for link in memory_links:
        if not isinstance(link, Mapping):
            continue
        unit_id = str(link.get("unit_id") or "").strip()
        if not unit_id:
            continue
        rec = _entry(unit_id)
        rec["memory_id"] = str(link.get("memory_id") or "") or None
        rec["memory_status"] = str(link.get("status") or "") or None

    for doc in graphrag_docs:
        if not isinstance(doc, Mapping):
            continue
        unit_id = str(doc.get("unit_id") or "").strip()
        if not unit_id:
            continue
        rec = _entry(unit_id)
        rec["graphrag_document_path"] = str(doc.get("document_path") or "") or None
        rec["qualified_name"] = str(doc.get("qualified_name") or "") or None
        rec["language"] = str(doc.get("language") or "") or None
        rec["unit_type"] = str(doc.get("unit_type") or "") or None
        rec["source_file_path"] = str(doc.get("source_file_path") or "") or None

    return sorted(by_unit.values(), key=lambda row: (str(row.get("qualified_name") or ""), str(row.get("unit_id") or "")))


def _extract_drift_highlights(drift_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    drift = _safe_dict(drift_payload)
    comparison = _safe_dict(drift.get("comparison"))
    comparisons = [row for row in _safe_list(comparison.get("comparisons")) if isinstance(row, Mapping)]
    warnings = [str(w) for w in _safe_list(comparison.get("warnings")) if str(w)]

    max_abs_delta = 0.0
    top_changes: list[dict[str, Any]] = []
    for row in comparisons:
        try:
            delta = float(row.get("delta") or 0.0)
        except (TypeError, ValueError):
            delta = 0.0
        abs_delta = abs(delta)
        if abs_delta > max_abs_delta:
            max_abs_delta = abs_delta
    ranked = sorted(
        comparisons,
        key=lambda row: abs(float(row.get("delta") or 0.0)) if isinstance(row, Mapping) else 0.0,
        reverse=True,
    )
    for row in ranked[:10]:
        top_changes.append(
            {
                "metric": row.get("metric"),
                "current": row.get("current"),
                "previous": row.get("previous"),
                "delta": row.get("delta"),
                "delta_pct": row.get("delta_pct"),
            }
        )

    return {
        "warnings": warnings,
        "warning_count": len(warnings),
        "max_abs_delta": max_abs_delta,
        "top_changes": top_changes,
    }


def _extract_benchmark_summary(benchmark_payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(benchmark_payload, Mapping):
        return None
    ingestion = _safe_dict(benchmark_payload.get("ingestion"))
    search = _safe_dict(benchmark_payload.get("search"))
    graph = _safe_dict(benchmark_payload.get("graph"))
    gate = _safe_dict(benchmark_payload.get("gate"))
    return {
        "path": benchmark_payload.get("path"),
        "generated_at": benchmark_payload.get("generated_at"),
        "ingestion_units_per_s_mean": _safe_dict(ingestion.get("units_per_s")).get("mean"),
        "search_p95_ms": _safe_dict(search.get("latency_ms")).get("p95"),
        "graph_build_ms": graph.get("build_ms"),
        "gate_pass": gate.get("pass"),
        "gate_violations": _safe_list(gate.get("violations")),
    }


def build_provenance_registry(
    *,
    provenance_payload: Mapping[str, Any],
    stage_summary_payload: Mapping[str, Any] | None = None,
    drift_payload: Mapping[str, Any] | None = None,
    benchmark_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = _safe_dict(dict(provenance_payload))
    stage_summary = _safe_dict(stage_summary_payload)

    unit_links = _build_unit_link_index(provenance)
    benchmark_summary = _extract_benchmark_summary(benchmark_payload)
    drift_summary = _extract_drift_highlights(drift_payload)

    digest = hashlib.sha256()
    digest.update(str(provenance.get("provenance_id") or "").encode("utf-8"))
    digest.update(str(provenance.get("generated_at") or "").encode("utf-8"))
    digest.update(str(len(unit_links)).encode("utf-8"))
    digest.update(str(benchmark_summary or {}).encode("utf-8"))
    digest.update(str(drift_summary).encode("utf-8"))

    record = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "registry_id": digest.hexdigest()[:24],
        "provenance_id": provenance.get("provenance_id"),
        "stage": provenance.get("stage"),
        "root_path": provenance.get("root_path"),
        "integration_policy": provenance.get("integration_policy"),
        "integration_run_id": provenance.get("integration_run_id"),
        "source_run_id": provenance.get("source_run_id"),
        "artifact_count": len(_safe_list(provenance.get("artifacts"))),
        "artifacts": _safe_list(provenance.get("artifacts")),
        "links": {
            "knowledge_count": int(_safe_dict(provenance.get("knowledge_links")).get("count") or 0),
            "memory_count": int(_safe_dict(provenance.get("memory_links")).get("count") or 0),
            "graphrag_count": int(_safe_dict(provenance.get("graphrag_links")).get("count") or 0),
            "unit_link_count": len(unit_links),
            "unit_links": unit_links,
        },
        "drift": drift_summary,
        "benchmark": benchmark_summary,
        "stage_summary_path": stage_summary.get("summary_path") or stage_summary.get("roundtrip_summary_path") or stage_summary.get("provenance_path"),
        "stage_summary": {
            "validation_pass": _safe_dict(stage_summary.get("validation")).get("pass"),
            "files_processed": _safe_dict(stage_summary.get("ingestion_stats")).get("files_processed"),
            "units_created": _safe_dict(stage_summary.get("ingestion_stats")).get("units_created"),
            "relationship_counts": stage_summary.get("relationship_counts"),
        },
    }
    return record


def write_provenance_registry(
    *,
    output_path: Path,
    provenance_payload: Mapping[str, Any],
    stage_summary_payload: Mapping[str, Any] | None = None,
    drift_payload: Mapping[str, Any] | None = None,
    benchmark_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = build_provenance_registry(
        provenance_payload=provenance_payload,
        stage_summary_payload=stage_summary_payload,
        drift_payload=drift_payload,
        benchmark_payload=benchmark_payload,
    )
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["path"] = str(output_path)
    return record


def read_provenance_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid provenance registry payload: {path}")
    return payload


def load_latest_benchmark_for_root(*, root_path: Path, search_roots: Iterable[Path]) -> dict[str, Any] | None:
    root = str(Path(root_path).resolve())
    candidates: list[Path] = []
    for base in search_roots:
        base = Path(base).resolve()
        if not base.exists():
            continue
        candidates.extend(base.rglob("*code_forge*benchmark*.json"))
        candidates.extend(base.rglob("code_forge_benchmark*.json"))

    newest: tuple[float, dict[str, Any]] | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        config = _safe_dict(payload.get("config"))
        candidate_root = str(Path(config.get("root_path") or "").resolve()) if config.get("root_path") else ""
        if candidate_root and candidate_root != root:
            continue
        mtime = candidate.stat().st_mtime
        payload = dict(payload)
        payload["path"] = str(candidate.resolve())
        if newest is None or mtime > newest[0]:
            newest = (mtime, payload)
    return newest[1] if newest else None
