from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from code_forge.analyzer.generic_analyzer import GenericCodeAnalyzer
from code_forge.ingest.runner import IngestionRunner
from code_forge.integration.pipeline import export_units_for_graphrag, sync_units_to_knowledge_forge
from code_forge.library.db import CodeLibraryDB


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return 0


def _should_skip(path: Path, exclude_patterns: Iterable[str]) -> bool:
    full = str(path)
    parts = set(path.parts)
    for pat in exclude_patterns:
        if not pat:
            continue
        if "/" in pat:
            if pat in full:
                return True
            continue
        if pat in parts:
            return True
    return False


def _category_for_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    suffix = path.suffix.lower()

    if any(p in {"test", "tests", "spec", "specs"} for p in parts) or path.name.startswith("test_"):
        return "test"
    if suffix in {".md", ".rst", ".txt", ".adoc"}:
        return "doc"
    if suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".conf"}:
        return "config"
    if suffix in {".sh", ".bash", ".zsh", ".ps1"}:
        return "script"
    if suffix in {".sql"}:
        return "data"
    return "source"


def build_repo_index(
    root_path: Path,
    output_dir: Path,
    *,
    extensions: Optional[Iterable[str]] = None,
    exclude_patterns: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
) -> dict[str, Any]:
    root_path = Path(root_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extension_set = {e.lower() for e in (extensions or GenericCodeAnalyzer.supported_extensions())}
    exclude = list(
        exclude_patterns
        or [
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            "dist",
            "build",
            ".mypy_cache",
            ".pytest_cache",
            "data/code_forge/graphrag_input",
            "doc_forge/final_docs",
        ]
    )

    entries: list[dict[str, Any]] = []
    by_language: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_extension: dict[str, int] = {}

    seen = 0
    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue
        if _should_skip(file_path, exclude):
            continue

        suffix = file_path.suffix.lower()
        if suffix not in extension_set:
            continue

        try:
            stat = file_path.stat()
        except OSError:
            continue

        rel_path = file_path.relative_to(root_path)
        language = GenericCodeAnalyzer.detect_language(file_path)
        category = _category_for_path(rel_path)

        entry = {
            "path": str(rel_path),
            "absolute_path": str(file_path),
            "language": language,
            "extension": suffix,
            "category": category,
            "bytes": int(stat.st_size),
            "line_count": _line_count(file_path),
            "sha256": _sha256_file(file_path),
            "modified_ts": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
        entries.append(entry)

        by_language[language] = by_language.get(language, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
        by_extension[suffix] = by_extension.get(suffix, 0) + 1

        seen += 1
        if max_files is not None and seen >= max_files:
            break

    payload = {
        "generated_at": _utc_now(),
        "root_path": str(root_path),
        "extensions": sorted(extension_set),
        "exclude_patterns": exclude,
        "files_total": len(entries),
        "by_language": dict(sorted(by_language.items(), key=lambda x: x[0])),
        "by_category": dict(sorted(by_category.items(), key=lambda x: x[0])),
        "by_extension": dict(sorted(by_extension.items(), key=lambda x: x[0])),
        "entries": entries,
    }

    (output_dir / "repo_index.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _count_file_hits_from_groups(groups: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for group in groups:
        units = group.get("units", [])
        if not isinstance(units, list):
            continue
        for unit in units:
            path = str((unit or {}).get("file_path") or "")
            if not path:
                continue
            out[path] = out.get(path, 0) + 1
    return out


def _count_file_hits_from_pairs(pairs: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for pair in pairs:
        left = pair.get("left") or {}
        right = pair.get("right") or {}
        for rec in (left, right):
            path = str(rec.get("file_path") or "")
            if not path:
                continue
            out[path] = out.get(path, 0) + 1
    return out


def build_duplication_index(
    db: CodeLibraryDB,
    output_dir: Path,
    *,
    min_occurrences: int = 2,
    limit_groups: int = 400,
    near_limit: int = 500,
    near_max_hamming: int = 6,
    near_min_tokens: int = 20,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exact = db.list_duplicate_units(
        min_occurrences=max(2, int(min_occurrences)),
        limit_groups=max(1, int(limit_groups)),
    )
    normalized = db.list_normalized_duplicates(
        min_occurrences=max(2, int(min_occurrences)),
        limit_groups=max(1, int(limit_groups)),
    )
    near_pairs = db.list_near_duplicates(
        max_hamming=max(0, int(near_max_hamming)),
        min_token_count=max(0, int(near_min_tokens)),
        limit_pairs=max(1, int(near_limit)),
    )

    exact_hits = _count_file_hits_from_groups(exact)
    normalized_hits = _count_file_hits_from_groups(normalized)
    near_hits = _count_file_hits_from_pairs(near_pairs)

    payload = {
        "generated_at": _utc_now(),
        "exact_duplicate_groups": exact,
        "normalized_duplicate_groups": normalized,
        "near_duplicate_pairs": near_pairs,
        "exact_hits_by_file": exact_hits,
        "normalized_hits_by_file": normalized_hits,
        "near_hits_by_file": near_hits,
        "summary": {
            "exact_group_count": len(exact),
            "normalized_group_count": len(normalized),
            "near_pair_count": len(near_pairs),
            "files_with_exact": len(exact_hits),
            "files_with_normalized": len(normalized_hits),
            "files_with_near": len(near_hits),
        },
    }

    (output_dir / "duplication_index.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _classify_file(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    unit_count = int(metrics.get("unit_count") or 0)
    duplicate_pressure = float(metrics.get("duplicate_pressure") or 0.0)
    avg_complexity = float(metrics.get("avg_complexity") or 0.0)
    max_complexity = float(metrics.get("max_complexity") or 0.0)
    callable_units = int(metrics.get("callable_units") or 0)
    is_test = bool(metrics.get("is_test"))

    if unit_count <= 0:
        reasons.append("No indexed code units detected for file")
        return "quarantine", reasons

    if is_test:
        reasons.append("Test file retained for behavioral proof and regression gates")
        return "keep", reasons

    if duplicate_pressure >= 2.0 and unit_count <= 8:
        reasons.append(f"High duplication pressure ({duplicate_pressure:.2f}) with small unit surface")
        reasons.append("Candidate for deletion after extraction of unique capability")
        return "delete_candidate", reasons

    if duplicate_pressure >= 1.0:
        reasons.append(f"Elevated duplication pressure ({duplicate_pressure:.2f})")
        reasons.append("Retain behavior but extract into canonical reusable module")
        return "extract", reasons

    if max_complexity >= 12.0 or avg_complexity >= 6.0:
        reasons.append(
            f"Complexity hotspot detected (avg={avg_complexity:.2f}, max={max_complexity:.2f})"
        )
        reasons.append("Refactor for modularity and testability")
        return "refactor", reasons

    if callable_units >= 2:
        reasons.append("Contains multiple callable symbols and low duplication")
        return "keep", reasons

    reasons.append("Insufficient evidence for direct keep/delete; hold for manual review")
    return "quarantine", reasons


def build_triage_report(
    db: CodeLibraryDB,
    repo_index: dict[str, Any],
    duplication_index: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_metrics = db.file_metrics(limit=max(1000, int(repo_index.get("files_total") or 0) * 4))
    metric_by_path = {str(rec.get("file_path")): rec for rec in file_metrics}

    exact_hits = {str(k): int(v) for k, v in (duplication_index.get("exact_hits_by_file") or {}).items()}
    normalized_hits = {
        str(k): int(v) for k, v in (duplication_index.get("normalized_hits_by_file") or {}).items()
    }
    near_hits = {str(k): int(v) for k, v in (duplication_index.get("near_hits_by_file") or {}).items()}

    entries: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}

    for item in repo_index.get("entries", []):
        rel_path = str(item.get("path") or "")
        if not rel_path:
            continue

        dbm = metric_by_path.get(rel_path, {})
        unit_count = int(dbm.get("unit_count") or 0)

        metrics = {
            "file_path": rel_path,
            "language": item.get("language"),
            "category": item.get("category"),
            "bytes": int(item.get("bytes") or 0),
            "line_count": int(item.get("line_count") or 0),
            "is_test": bool(item.get("category") == "test"),
            "unit_count": unit_count,
            "module_units": int(dbm.get("module_units") or 0),
            "class_units": int(dbm.get("class_units") or 0),
            "callable_units": int(dbm.get("callable_units") or 0),
            "avg_complexity": float(dbm.get("avg_complexity") or 0.0),
            "max_complexity": float(dbm.get("max_complexity") or 0.0),
            "token_count_sum": int(dbm.get("token_count_sum") or 0),
            "unique_fingerprint_count": int(dbm.get("unique_fingerprint_count") or 0),
            "exact_duplicate_hits": exact_hits.get(rel_path, 0),
            "normalized_duplicate_hits": normalized_hits.get(rel_path, 0),
            "near_duplicate_hits": near_hits.get(rel_path, 0),
        }
        metrics["duplicate_pressure"] = (
            metrics["exact_duplicate_hits"]
            + metrics["normalized_duplicate_hits"]
            + metrics["near_duplicate_hits"]
        ) / max(1, unit_count)

        label, reasons = _classify_file(metrics)
        label_counts[label] = label_counts.get(label, 0) + 1

        entries.append(
            {
                "file_path": rel_path,
                "label": label,
                "reasons": reasons,
                "metrics": metrics,
            }
        )

    entries.sort(key=lambda rec: (rec["label"], rec["file_path"]))

    csv_path = output_dir / "triage.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file_path",
                "label",
                "language",
                "category",
                "unit_count",
                "duplicate_pressure",
                "avg_complexity",
                "max_complexity",
                "exact_duplicate_hits",
                "normalized_duplicate_hits",
                "near_duplicate_hits",
                "reason_1",
                "reason_2",
            ]
        )
        for rec in entries:
            metrics = rec["metrics"]
            reasons = rec.get("reasons", [])
            writer.writerow(
                [
                    rec["file_path"],
                    rec["label"],
                    metrics.get("language"),
                    metrics.get("category"),
                    metrics.get("unit_count"),
                    round(float(metrics.get("duplicate_pressure") or 0.0), 4),
                    round(float(metrics.get("avg_complexity") or 0.0), 4),
                    round(float(metrics.get("max_complexity") or 0.0), 4),
                    metrics.get("exact_duplicate_hits"),
                    metrics.get("normalized_duplicate_hits"),
                    metrics.get("near_duplicate_hits"),
                    reasons[0] if len(reasons) > 0 else "",
                    reasons[1] if len(reasons) > 1 else "",
                ]
            )

    report_lines = [
        "# Code Forge Triage Report",
        "",
        f"Generated: {_utc_now()}",
        f"Scanned files: {len(entries)}",
        "",
        "## Label Distribution",
        "",
    ]
    for label in sorted(label_counts):
        report_lines.append(f"- `{label}`: {label_counts[label]}")

    report_lines.append("")
    report_lines.append("## Highest Duplication Pressure")
    report_lines.append("")

    top_dup = sorted(entries, key=lambda rec: rec["metrics"].get("duplicate_pressure", 0.0), reverse=True)[:25]
    for rec in top_dup:
        m = rec["metrics"]
        report_lines.append(
            f"- `{rec['file_path']}` | label=`{rec['label']}` | duplicate_pressure={m.get('duplicate_pressure', 0.0):.2f} | units={m.get('unit_count', 0)}"
        )

    (output_dir / "triage_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    payload = {
        "generated_at": _utc_now(),
        "entries": entries,
        "label_counts": label_counts,
        "csv_path": str(csv_path),
        "report_path": str(output_dir / "triage_report.md"),
    }
    (output_dir / "triage.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_archive_digester(
    *,
    root_path: Path,
    db: CodeLibraryDB,
    runner: IngestionRunner,
    output_dir: Path,
    mode: str = "analysis",
    extensions: Optional[Iterable[str]] = None,
    exclude_patterns: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    progress_every: int = 200,
    sync_knowledge_path: Optional[Path] = None,
    graphrag_output_dir: Optional[Path] = None,
    graph_export_limit: int = 20000,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = runner.ingest_path(
        Path(root_path),
        mode=mode,
        extensions=extensions,
        exclude_patterns=list(exclude_patterns) if exclude_patterns else None,
        max_files=max_files,
        progress_every=max(1, int(progress_every)),
    )

    repo_index = build_repo_index(
        root_path=Path(root_path),
        output_dir=output_dir,
        extensions=extensions,
        exclude_patterns=exclude_patterns,
        max_files=max_files,
    )
    duplication = build_duplication_index(db=db, output_dir=output_dir)
    triage = build_triage_report(db=db, repo_index=repo_index, duplication_index=duplication, output_dir=output_dir)

    knowledge_sync = None
    if sync_knowledge_path is not None:
        knowledge_sync = sync_units_to_knowledge_forge(
            db=db,
            kb_path=Path(sync_knowledge_path),
            limit=max(1, int(graph_export_limit)),
            min_token_count=5,
        )

    graphrag_export = None
    if graphrag_output_dir is not None:
        graphrag_export = export_units_for_graphrag(
            db=db,
            output_dir=Path(graphrag_output_dir),
            limit=max(1, int(graph_export_limit)),
            min_token_count=5,
        )

    summary = {
        "generated_at": _utc_now(),
        "root_path": str(Path(root_path).resolve()),
        "output_dir": str(output_dir),
        "ingestion_stats": asdict(stats),
        "repo_index_path": str(output_dir / "repo_index.json"),
        "duplication_index_path": str(output_dir / "duplication_index.json"),
        "triage_json_path": str(output_dir / "triage.json"),
        "triage_report_path": str(output_dir / "triage_report.md"),
        "knowledge_sync": knowledge_sync,
        "graphrag_export": graphrag_export,
    }
    (output_dir / "archive_digester_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
