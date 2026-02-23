from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FILES = {
    "repo_index.json": "repo_index",
    "duplication_index.json": "duplication_index",
    "dependency_graph.json": "dependency_graph",
    "triage.json": "triage",
    "triage_audit.json": "triage_audit",
    "archive_digester_summary.json": "archive_summary",
}

OPTIONAL_FILES = {
    "drift_report.json": "drift_report",
    "provenance_links.json": "provenance_links",
    "provenance_registry.json": "provenance_registry",
    "archive_reduction_plan.json": "archive_reduction_plan",
}


def _require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate_repo_index(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "repo_index.generated_at must be a string")
    _require(isinstance(payload.get("root_path"), str), errors, "repo_index.root_path must be a string")
    _require(isinstance(payload.get("entries"), list), errors, "repo_index.entries must be a list")
    _require(isinstance(payload.get("files_total"), int), errors, "repo_index.files_total must be an int")

    entries = payload.get("entries") or []
    for idx, entry in enumerate(entries[:50]):
        prefix = f"repo_index.entries[{idx}]"
        _require(isinstance(entry, dict), errors, f"{prefix} must be an object")
        if not isinstance(entry, dict):
            continue
        for key in ("path", "language", "extension", "category", "sha256"):
            _require(isinstance(entry.get(key), str), errors, f"{prefix}.{key} must be a string")
    return errors


def validate_duplication_index(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "duplication_index.generated_at must be a string")
    for key in (
        "exact_duplicate_groups",
        "normalized_duplicate_groups",
        "structural_duplicate_groups",
        "near_duplicate_pairs",
    ):
        _require(isinstance(payload.get(key), list), errors, f"duplication_index.{key} must be a list")
    _require(isinstance(payload.get("summary"), dict), errors, "duplication_index.summary must be an object")
    return errors


def validate_dependency_graph(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "dependency_graph.generated_at must be a string")
    _require(isinstance(payload.get("nodes"), list), errors, "dependency_graph.nodes must be a list")
    _require(isinstance(payload.get("edges"), list), errors, "dependency_graph.edges must be a list")
    _require(isinstance(payload.get("summary"), dict), errors, "dependency_graph.summary must be an object")

    for idx, edge in enumerate((payload.get("edges") or [])[:100]):
        prefix = f"dependency_graph.edges[{idx}]"
        _require(isinstance(edge, dict), errors, f"{prefix} must be an object")
        if not isinstance(edge, dict):
            continue
        for key in ("source", "target", "rel_type"):
            _require(isinstance(edge.get(key), str), errors, f"{prefix}.{key} must be a string")
        _require(isinstance(edge.get("weight"), int), errors, f"{prefix}.weight must be an int")
    return errors


def validate_triage(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "triage.generated_at must be a string")
    _require(isinstance(payload.get("entries"), list), errors, "triage.entries must be a list")
    _require(isinstance(payload.get("label_counts"), dict), errors, "triage.label_counts must be an object")

    entries = payload.get("entries") or []
    for idx, entry in enumerate(entries[:100]):
        prefix = f"triage.entries[{idx}]"
        _require(isinstance(entry, dict), errors, f"{prefix} must be an object")
        if not isinstance(entry, dict):
            continue
        _require(isinstance(entry.get("file_path"), str), errors, f"{prefix}.file_path must be a string")
        _require(isinstance(entry.get("label"), str), errors, f"{prefix}.label must be a string")
        _require(isinstance(entry.get("confidence"), (int, float)), errors, f"{prefix}.confidence must be numeric")
        _require(isinstance(entry.get("rule_id"), str), errors, f"{prefix}.rule_id must be a string")
        _require(isinstance(entry.get("metrics"), dict), errors, f"{prefix}.metrics must be an object")
        _require(isinstance(entry.get("reasons"), list), errors, f"{prefix}.reasons must be a list")
    return errors


def validate_triage_audit(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "triage_audit.generated_at must be a string")
    _require(isinstance(payload.get("ruleset_version"), str), errors, "triage_audit.ruleset_version must be string")
    _require(isinstance(payload.get("thresholds"), dict), errors, "triage_audit.thresholds must be object")
    _require(isinstance(payload.get("decisions"), list), errors, "triage_audit.decisions must be list")
    for idx, decision in enumerate((payload.get("decisions") or [])[:100]):
        prefix = f"triage_audit.decisions[{idx}]"
        _require(isinstance(decision, dict), errors, f"{prefix} must be object")
        if not isinstance(decision, dict):
            continue
        _require(isinstance(decision.get("file_path"), str), errors, f"{prefix}.file_path must be string")
        _require(isinstance(decision.get("rule_id"), str), errors, f"{prefix}.rule_id must be string")
        _require(isinstance(decision.get("label"), str), errors, f"{prefix}.label must be string")
        _require(isinstance(decision.get("confidence"), (int, float)), errors, f"{prefix}.confidence must be numeric")
    return errors


def validate_archive_summary(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "archive_summary.generated_at must be a string")
    _require(isinstance(payload.get("ingestion_stats"), dict), errors, "archive_summary.ingestion_stats must be object")
    _require(
        isinstance(payload.get("relationship_counts"), dict),
        errors,
        "archive_summary.relationship_counts must be object",
    )
    for key in (
        "repo_index_path",
        "duplication_index_path",
        "dependency_graph_path",
        "triage_json_path",
        "triage_audit_path",
    ):
        _require(isinstance(payload.get(key), str), errors, f"archive_summary.{key} must be a string")
    validation = payload.get("validation")
    if validation is not None:
        _require(isinstance(validation, dict), errors, "archive_summary.validation must be object")
        if isinstance(validation, dict):
            _require(isinstance(validation.get("pass"), bool), errors, "archive_summary.validation.pass must be bool")
    if payload.get("provenance_registry_path") is not None:
        _require(
            isinstance(payload.get("provenance_registry_path"), str),
            errors,
            "archive_summary.provenance_registry_path must be a string when present",
        )
    return errors


def validate_drift_report(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "drift_report.generated_at must be a string")
    _require(isinstance(payload.get("output_dir"), str), errors, "drift_report.output_dir must be a string")
    _require(
        isinstance(payload.get("current_snapshot_path"), str),
        errors,
        "drift_report.current_snapshot_path must be a string",
    )
    comparison = payload.get("comparison")
    _require(isinstance(comparison, dict), errors, "drift_report.comparison must be an object")
    if isinstance(comparison, dict):
        _require(
            isinstance(comparison.get("compared_metric_count"), int),
            errors,
            "drift_report.comparison.compared_metric_count must be an int",
        )
        _require(
            isinstance(comparison.get("comparisons"), list),
            errors,
            "drift_report.comparison.comparisons must be a list",
        )
        _require(
            isinstance(comparison.get("warnings"), list),
            errors,
            "drift_report.comparison.warnings must be a list",
        )
    return errors


def validate_provenance_links(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "provenance.generated_at must be a string")
    _require(isinstance(payload.get("stage"), str), errors, "provenance.stage must be a string")
    _require(isinstance(payload.get("root_path"), str), errors, "provenance.root_path must be a string")
    _require(
        isinstance(payload.get("integration_policy"), str),
        errors,
        "provenance.integration_policy must be a string",
    )
    _require(
        payload.get("integration_run_id") is None or isinstance(payload.get("integration_run_id"), str),
        errors,
        "provenance.integration_run_id must be string or null",
    )
    _require(isinstance(payload.get("artifacts"), list), errors, "provenance.artifacts must be a list")
    _require(isinstance(payload.get("knowledge_links"), dict), errors, "provenance.knowledge_links must be object")
    _require(isinstance(payload.get("memory_links"), dict), errors, "provenance.memory_links must be object")
    _require(isinstance(payload.get("graphrag_links"), dict), errors, "provenance.graphrag_links must be object")
    return errors


def validate_provenance_registry(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(
        isinstance(payload.get("schema_version"), str), errors, "provenance_registry.schema_version must be string"
    )
    _require(isinstance(payload.get("generated_at"), str), errors, "provenance_registry.generated_at must be string")
    _require(isinstance(payload.get("registry_id"), str), errors, "provenance_registry.registry_id must be string")
    _require(isinstance(payload.get("provenance_id"), str), errors, "provenance_registry.provenance_id must be string")
    _require(isinstance(payload.get("stage"), str), errors, "provenance_registry.stage must be string")
    _require(isinstance(payload.get("root_path"), str), errors, "provenance_registry.root_path must be string")
    _require(isinstance(payload.get("links"), dict), errors, "provenance_registry.links must be object")
    _require(isinstance(payload.get("drift"), dict), errors, "provenance_registry.drift must be object")
    if payload.get("benchmark") is not None:
        _require(isinstance(payload.get("benchmark"), dict), errors, "provenance_registry.benchmark must be object")
    links = payload.get("links") if isinstance(payload.get("links"), dict) else {}
    _require(
        isinstance(links.get("unit_links"), list),
        errors,
        "provenance_registry.links.unit_links must be list",
    )
    for idx, row in enumerate((links.get("unit_links") or [])[:100]):
        prefix = f"provenance_registry.links.unit_links[{idx}]"
        _require(isinstance(row, dict), errors, f"{prefix} must be object")
        if not isinstance(row, dict):
            continue
        _require(isinstance(row.get("unit_id"), str), errors, f"{prefix}.unit_id must be string")
    return errors


def validate_archive_reduction_plan(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(
        isinstance(payload.get("generated_at"), str),
        errors,
        "archive_reduction_plan.generated_at must be a string",
    )
    _require(
        isinstance(payload.get("source_triage_path"), str),
        errors,
        "archive_reduction_plan.source_triage_path must be a string",
    )
    _require(isinstance(payload.get("counts"), dict), errors, "archive_reduction_plan.counts must be an object")
    for key in ("delete_candidates", "extract_candidates", "refactor_candidates", "quarantine_candidates"):
        _require(
            isinstance(payload.get(key), list),
            errors,
            f"archive_reduction_plan.{key} must be a list",
        )
    return errors


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def validate_output_dir(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    report: dict[str, Any] = {
        "output_dir": str(output_dir.resolve()),
        "files": {},
        "pass": True,
        "errors": [],
    }

    validators = {
        "repo_index": validate_repo_index,
        "duplication_index": validate_duplication_index,
        "dependency_graph": validate_dependency_graph,
        "triage": validate_triage,
        "triage_audit": validate_triage_audit,
        "archive_summary": validate_archive_summary,
        "drift_report": validate_drift_report,
        "provenance_links": validate_provenance_links,
        "provenance_registry": validate_provenance_registry,
        "archive_reduction_plan": validate_archive_reduction_plan,
    }

    for filename, key in REQUIRED_FILES.items():
        path = output_dir / filename
        file_report: dict[str, Any] = {"path": str(path), "exists": path.exists(), "errors": []}
        report["files"][filename] = file_report
        if not path.exists():
            file_report["errors"].append("missing required artifact")
            report["errors"].append(f"{filename}: missing required artifact")
            report["pass"] = False
            continue

        try:
            payload = _load_json(path)
            validate_errors = validators[key](payload)
            file_report["errors"].extend(validate_errors)
            if validate_errors:
                report["errors"].extend([f"{filename}: {e}" for e in validate_errors])
                report["pass"] = False
        except Exception as exc:
            message = f"failed to load/validate JSON: {exc}"
            file_report["errors"].append(message)
            report["errors"].append(f"{filename}: {message}")
            report["pass"] = False

    for filename, key in OPTIONAL_FILES.items():
        path = output_dir / filename
        if not path.exists():
            continue
        file_report: dict[str, Any] = {"path": str(path), "exists": True, "errors": []}
        report["files"][filename] = file_report
        try:
            payload = _load_json(path)
            validate_errors = validators[key](payload)
            file_report["errors"].extend(validate_errors)
            if validate_errors:
                report["errors"].extend([f"{filename}: {e}" for e in validate_errors])
                report["pass"] = False
        except Exception as exc:
            message = f"failed to load/validate JSON: {exc}"
            file_report["errors"].append(message)
            report["errors"].append(f"{filename}: {message}")
            report["pass"] = False

    return report
