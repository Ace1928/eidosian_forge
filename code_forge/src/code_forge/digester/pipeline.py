from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from code_forge.analyzer.generic_analyzer import GenericCodeAnalyzer
from code_forge.digester.drift import build_drift_report_from_output
from code_forge.digester.schema import validate_output_dir
from code_forge.ingest.runner import IngestionRunner
from code_forge.integration.memory import sync_units_to_memory_forge
from code_forge.integration.pipeline import export_units_for_graphrag, sync_units_to_knowledge_forge
from code_forge.integration.provenance import write_provenance_links
from code_forge.integration.provenance_registry import (
    load_latest_benchmark_for_root,
    write_provenance_registry,
)
from code_forge.library.db import CodeLibraryDB

TRIAGE_RULESET_VERSION = "code_forge_triage_ruleset_v3_2026_02_23"
TRIAGE_THRESHOLDS = {
    "unit_count_min": 1,
    "delete_candidate_duplicate_pressure": 2.0,
    "delete_candidate_unit_count_max": 8,
    "extract_duplicate_pressure": 1.0,
    "refactor_max_complexity": 12.0,
    "refactor_avg_complexity": 6.0,
    "keep_callable_units": 2,
    "hot_path_preserve_threshold": 0.35,
    "hot_path_keep_threshold": 1.0,
}


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


def _normalize_include_paths(include_paths: Optional[Iterable[str]]) -> set[str]:
    return {
        str(Path(item)).replace("\\", "/").lstrip("./")
        for item in (include_paths or [])
        if str(item).strip()
    }


def build_repo_index(
    root_path: Path,
    output_dir: Path,
    *,
    extensions: Optional[Iterable[str]] = None,
    exclude_patterns: Optional[Iterable[str]] = None,
    include_paths: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
) -> dict[str, Any]:
    root_path = Path(root_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extension_set = {e.lower() for e in (extensions or GenericCodeAnalyzer.supported_extensions())}
    include_set = _normalize_include_paths(include_paths)
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
    candidate_paths: Iterable[Path]
    if include_set:
        candidate_paths = [(root_path / rel_path).resolve() for rel_path in sorted(include_set)]
    else:
        candidate_paths = root_path.rglob("*")

    for file_path in candidate_paths:
        if not file_path.is_file():
            continue
        if _should_skip(file_path, exclude):
            continue

        rel_path = file_path.relative_to(root_path)
        rel_path_str = str(rel_path).replace("\\", "/")

        suffix = file_path.suffix.lower()
        if suffix not in extension_set:
            continue

        try:
            stat = file_path.stat()
        except OSError:
            continue

        language = GenericCodeAnalyzer.detect_language(file_path)
        category = _category_for_path(rel_path)

        entry = {
            "path": rel_path_str,
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
        "include_paths": sorted(include_set),
        "exclude_patterns": exclude,
        "files_total": len(entries),
        "by_language": dict(sorted(by_language.items(), key=lambda x: x[0])),
        "by_category": dict(sorted(by_category.items(), key=lambda x: x[0])),
        "by_extension": dict(sorted(by_extension.items(), key=lambda x: x[0])),
        "entries": entries,
    }

    (output_dir / "repo_index.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _route_for_entry(entry: dict[str, Any]) -> str:
    category = str(entry.get("category") or "")
    extension = str(entry.get("extension") or "").lower()
    language = str(entry.get("language") or "").lower()

    if extension in {".md", ".rst", ".txt", ".adoc", ".pdf", ".html", ".htm", ".ipynb"}:
        return "document_pipeline"
    if extension in {".json", ".yaml", ".yml", ".toml", ".ini", ".csv", ".tsv", ".xml"}:
        return "knowledge_metadata"
    if extension in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg", ".mp3", ".wav", ".mp4", ".mov"}:
        return "defer_binary"
    if category in {"source", "script", "test"} or language not in {"unknown", "", "text"}:
        return "code_forge"
    return "manual_review"


def build_archive_ingestion_batches(
    repo_index: dict[str, Any],
    output_dir: Path,
    *,
    max_files_per_batch: int = 500,
    max_bytes_per_batch: int = 50_000_000,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in list(repo_index.get("entries") or []):
        if not isinstance(entry, dict):
            continue
        route = _route_for_entry(entry)
        category = str(entry.get("category") or "unknown")
        route_groups.setdefault((route, category), []).append(entry)

    batches: list[dict[str, Any]] = []
    for (route, category), entries in sorted(route_groups.items()):
        ordered = sorted(entries, key=lambda row: (str(row.get("extension") or ""), str(row.get("path") or "")))
        chunk: list[dict[str, Any]] = []
        chunk_bytes = 0
        chunk_no = 1

        def _flush() -> None:
            nonlocal chunk, chunk_bytes, chunk_no
            if not chunk:
                return
            batch_files = [str(item.get("path") or "") for item in chunk]
            batch_id = hashlib.sha256(
                f"{route}|{category}|{chunk_no}|{'|'.join(batch_files)}".encode("utf-8")
            ).hexdigest()[:16]
            batches.append(
                {
                    "batch_id": batch_id,
                    "route": route,
                    "category": category,
                    "sequence": chunk_no,
                    "file_count": len(chunk),
                    "total_bytes": chunk_bytes,
                    "extensions": sorted({str(item.get("extension") or "") for item in chunk}),
                    "paths": batch_files,
                    "status": "pending",
                }
            )
            chunk = []
            chunk_bytes = 0
            chunk_no += 1

        for entry in ordered:
            entry_bytes = int(entry.get("bytes") or 0)
            if chunk and (
                len(chunk) >= max(1, int(max_files_per_batch)) or (chunk_bytes + entry_bytes) > max_bytes_per_batch
            ):
                _flush()
            chunk.append(entry)
            chunk_bytes += entry_bytes
        _flush()

    payload = {
        "generated_at": _utc_now(),
        "root_path": repo_index.get("root_path"),
        "files_total": int(repo_index.get("files_total") or 0),
        "batch_count": len(batches),
        "route_counts": {
            route: sum(1 for batch in batches if batch["route"] == route)
            for route in sorted({batch["route"] for batch in batches})
        },
        "batches": batches,
    }
    (output_dir / "archive_ingestion_batches.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def initialize_archive_ingestion_state(
    batch_plan: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batches = list(batch_plan.get("batches") or [])
    payload = {
        "generated_at": _utc_now(),
        "root_path": batch_plan.get("root_path"),
        "batch_count": len(batches),
        "completed_count": 0,
        "batches": {
            str(batch["batch_id"]): {
                "route": batch.get("route"),
                "category": batch.get("category"),
                "status": "pending",
                "attempts": 0,
                "last_error": None,
                "updated_at": None,
            }
            for batch in batches
            if isinstance(batch, dict) and batch.get("batch_id")
        },
    }
    (output_dir / "archive_ingestion_state.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_archive_ingestion_state(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    state_path = output_dir / "archive_ingestion_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing archive ingestion state: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def update_archive_ingestion_state(
    output_dir: Path,
    *,
    batch_id: str,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    state_path = output_dir / "archive_ingestion_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing archive ingestion state: {state_path}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    batches = payload.get("batches") or {}
    if batch_id not in batches:
        raise KeyError(f"unknown batch_id: {batch_id}")
    batch = batches[batch_id]
    batch["status"] = str(status)
    batch["attempts"] = int(batch.get("attempts") or 0) + 1
    batch["last_error"] = error
    batch["updated_at"] = _utc_now()
    payload["completed_count"] = sum(1 for item in batches.values() if str(item.get("status")) == "completed")
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _batch_extensions(batch: dict[str, Any]) -> list[str]:
    return [
        str(ext)
        for ext in list(batch.get("extensions") or [])
        if str(ext).strip()
    ]


def _word_forge_seed_from_text(text: str) -> dict[str, Any]:
    try:
        from eidos_mcp.routers.word_forge import wf_build_from_text

        payload = wf_build_from_text(text, min_word_length=4)
        if isinstance(payload, str):
            return json.loads(payload)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    return {"status": "error", "error": "unexpected word_forge payload"}


def _load_tika_ingester(kb_path: Path):
    from crawl_forge.tika_extractor import TikaExtractor, TikaKnowledgeIngester
    from knowledge_forge.core.graph import KnowledgeForge

    tika = TikaExtractor(enable_cache=True)
    kb = KnowledgeForge(persistence_path=kb_path)
    return tika, TikaKnowledgeIngester(tika=tika, knowledge_forge=kb)


def _process_document_batch(
    *,
    root_path: Path,
    batch: dict[str, Any],
    output_dir: Path,
    kb_path: Path,
) -> dict[str, Any]:
    tika, ingester = _load_tika_ingester(kb_path)
    batch_id = str(batch.get("batch_id") or "")
    results: list[dict[str, Any]] = []
    files_processed = 0
    nodes_created = 0
    lexicon_updates = 0
    errors: list[dict[str, Any]] = []

    for rel_path in list(batch.get("paths") or []):
        source_path = Path(root_path) / str(rel_path)
        ingest_payload = ingester.ingest_file(
            source_path,
            tags=["archive_forge", "document_pipeline", f"archive_batch:{batch_id}"],
        )
        extract_payload = tika.extract_from_file(source_path)
        if ingest_payload.get("status") == "success":
            files_processed += 1
            nodes_created += int(ingest_payload.get("nodes_created") or 0)
        else:
            errors.append(
                {
                    "path": str(rel_path),
                    "error": ingest_payload.get("error") or ingest_payload.get("status") or "unknown_error",
                }
            )

        lexicon_payload = {"status": "skipped"}
        content = str(extract_payload.get("content") or "").strip()
        if content:
            lexicon_payload = _word_forge_seed_from_text(content[:12000])
            if lexicon_payload.get("status") == "success":
                lexicon_updates += int(lexicon_payload.get("nodes_added") or 0)

        results.append(
            {
                "path": str(rel_path),
                "ingest": ingest_payload,
                "extract_status": extract_payload.get("status"),
                "lexicon": lexicon_payload,
            }
        )

    payload = {
        "generated_at": _utc_now(),
        "batch_id": batch_id,
        "route": batch.get("route"),
        "files_processed": files_processed,
        "nodes_created": nodes_created,
        "lexicon_nodes_added": lexicon_updates,
        "results": results,
        "errors": errors,
    }
    (output_dir / "document_batch_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _process_metadata_batch(
    *,
    root_path: Path,
    batch: dict[str, Any],
    output_dir: Path,
    kb_path: Path,
) -> dict[str, Any]:
    from knowledge_forge.core.graph import KnowledgeForge

    kb = KnowledgeForge(persistence_path=kb_path)
    batch_id = str(batch.get("batch_id") or "")
    results: list[dict[str, Any]] = []
    files_processed = 0
    nodes_created = 0
    lexicon_updates = 0
    errors: list[dict[str, Any]] = []

    for rel_path in list(batch.get("paths") or []):
        source_path = Path(root_path) / str(rel_path)
        try:
            text = source_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            errors.append({"path": str(rel_path), "error": str(exc)})
            continue

        node = kb.add_knowledge(
            content=text[:8000],
            concepts=["archive_metadata", source_path.suffix.lstrip(".") or "metadata"],
            tags=[
                "archive_forge",
                "knowledge_metadata",
                f"archive_batch:{batch_id}",
                f"type:{source_path.suffix.lstrip('.') or 'unknown'}",
            ],
            metadata={
                "source": "archive_forge",
                "path": str(rel_path),
                "route": "knowledge_metadata",
                "batch_id": batch_id,
            },
        )
        files_processed += 1
        nodes_created += 1

        lexicon_payload = {"status": "skipped"}
        if text.strip():
            lexicon_payload = _word_forge_seed_from_text(text[:12000])
            if lexicon_payload.get("status") == "success":
                lexicon_updates += int(lexicon_payload.get("nodes_added") or 0)

        results.append(
            {
                "path": str(rel_path),
                "node_id": node.id,
                "lexicon": lexicon_payload,
            }
        )

    payload = {
        "generated_at": _utc_now(),
        "batch_id": batch_id,
        "route": batch.get("route"),
        "files_processed": files_processed,
        "nodes_created": nodes_created,
        "lexicon_nodes_added": lexicon_updates,
        "results": results,
        "errors": errors,
    }
    (output_dir / "metadata_batch_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_archive_ingestion_batches(
    *,
    root_path: Path,
    db: CodeLibraryDB,
    runner: IngestionRunner,
    output_dir: Path,
    kb_path: Path,
    graphrag_output_dir: Optional[Path] = None,
    batch_limit: Optional[int] = None,
    retry_failed: bool = False,
    include_routes: Optional[Iterable[str]] = None,
    extensions: Optional[Iterable[str]] = None,
    exclude_patterns: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    progress_every: int = 200,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "archive_ingestion_batches.json"
    state_path = output_dir / "archive_ingestion_state.json"

    if plan_path.exists():
        batch_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        repo_index = build_repo_index(
            root_path=Path(root_path),
            output_dir=output_dir,
            extensions=extensions,
            exclude_patterns=exclude_patterns,
            max_files=max_files,
        )
        batch_plan = build_archive_ingestion_batches(repo_index, output_dir)

    if state_path.exists():
        state = load_archive_ingestion_state(output_dir)
    else:
        state = initialize_archive_ingestion_state(batch_plan, output_dir)

    allowed_statuses = {"pending"}
    if retry_failed:
        allowed_statuses.add("failed")
    allowed_routes = {str(route) for route in (include_routes or []) if str(route).strip()}

    selected_batches: list[dict[str, Any]] = []
    for batch in list(batch_plan.get("batches") or []):
        if not isinstance(batch, dict):
            continue
        batch_id = str(batch.get("batch_id") or "")
        if not batch_id:
            continue
        batch_state = (state.get("batches") or {}).get(batch_id) or {}
        if str(batch_state.get("status") or "pending") not in allowed_statuses:
            continue
        if allowed_routes and str(batch.get("route") or "") not in allowed_routes:
            continue
        selected_batches.append(batch)

    if batch_limit is not None:
        selected_batches = selected_batches[: max(0, int(batch_limit))]

    runs: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    skipped = 0
    for batch in selected_batches:
        batch_id = str(batch.get("batch_id") or "")
        batch_route = str(batch.get("route") or "manual_review")
        batch_dir = output_dir / "batches" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        update_archive_ingestion_state(output_dir, batch_id=batch_id, status="in_progress")

        try:
            if batch_route == "code_forge":
                payload = run_archive_digester(
                    root_path=Path(root_path),
                    db=db,
                    runner=runner,
                    output_dir=batch_dir,
                    extensions=_batch_extensions(batch),
                    include_paths=list(batch.get("paths") or []),
                    progress_every=progress_every,
                    sync_knowledge_path=kb_path,
                    graphrag_output_dir=(Path(graphrag_output_dir) / batch_id) if graphrag_output_dir else None,
                )
                payload["batch_id"] = batch_id
                payload["route"] = batch_route
            elif batch_route == "document_pipeline":
                payload = _process_document_batch(
                    root_path=Path(root_path),
                    batch=batch,
                    output_dir=batch_dir,
                    kb_path=kb_path,
                )
            elif batch_route == "knowledge_metadata":
                payload = _process_metadata_batch(
                    root_path=Path(root_path),
                    batch=batch,
                    output_dir=batch_dir,
                    kb_path=kb_path,
                )
            else:
                payload = {
                    "generated_at": _utc_now(),
                    "batch_id": batch_id,
                    "route": batch_route,
                    "status": "deferred",
                    "reason": "route does not have an active ingestion executor yet",
                    "paths": list(batch.get("paths") or []),
                }
                (batch_dir / "batch_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                skipped += 1
                update_archive_ingestion_state(output_dir, batch_id=batch_id, status="skipped")
                runs.append(payload)
                continue

            (batch_dir / "batch_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            update_archive_ingestion_state(output_dir, batch_id=batch_id, status="completed")
            completed += 1
            runs.append(payload)
        except Exception as exc:
            failed += 1
            update_archive_ingestion_state(output_dir, batch_id=batch_id, status="failed", error=str(exc))
            runs.append(
                {
                    "generated_at": _utc_now(),
                    "batch_id": batch_id,
                    "route": batch_route,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    state = load_archive_ingestion_state(output_dir)
    summary = {
        "generated_at": _utc_now(),
        "root_path": str(Path(root_path).resolve()),
        "batch_plan_path": str(plan_path),
        "state_path": str(state_path),
        "selected_batches": len(selected_batches),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "runs": runs,
        "state": state,
    }
    (output_dir / "archive_ingestion_wave_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


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


def _normalize_profile_path(path: str, *, root_path: Path | None = None) -> str:
    text = str(path or "").strip().replace("\\", "/")
    if not text:
        return ""
    p = Path(text)
    if root_path is not None:
        try:
            return str(p.resolve().relative_to(root_path.resolve())).replace("\\", "/")
        except Exception:
            pass
    if text.startswith("./"):
        text = text[2:]
    return text


def load_profile_hotspots(
    profile_trace_path: Path | None,
    *,
    root_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Load optional profile hotness hints keyed by repository-relative file path.

    Supported JSON layouts:
    - {"file_hotness": {"path.py": 1.2, ...}}
    - {"hotspots": [{"file_path": "...", "hotness": 0.8, "samples": 4}, ...]}
    - {"entries": [{"path": "...", "duration_ms": 42.0, "calls": 3}, ...]}
    """

    if profile_trace_path is None:
        return {}
    path = Path(profile_trace_path).resolve()
    if not path.exists() or not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: dict[str, dict[str, Any]] = {}

    def _add(path_value: Any, *, hotness: Any, samples: Any = 1, source: str = "") -> None:
        rel = _normalize_profile_path(str(path_value or ""), root_path=root_path)
        if not rel:
            return
        try:
            hot = float(hotness or 0.0)
        except (TypeError, ValueError):
            hot = 0.0
        if hot <= 0.0:
            return
        try:
            count = max(1, int(samples or 1))
        except (TypeError, ValueError):
            count = 1
        current = out.get(rel)
        if current is None:
            out[rel] = {
                "hotness": hot,
                "samples": count,
                "source": source or path.name,
            }
            return
        current["hotness"] = float(current.get("hotness") or 0.0) + hot
        current["samples"] = int(current.get("samples") or 0) + count

    if isinstance(payload, dict):
        file_hotness = payload.get("file_hotness")
        if isinstance(file_hotness, dict):
            for rel, score in file_hotness.items():
                _add(rel, hotness=score, samples=1, source="file_hotness")

        for key in ("hotspots", "entries"):
            rows = payload.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rel = row.get("file_path") or row.get("path") or row.get("file")
                hot = row.get("hotness")
                if hot is None:
                    duration_ms = float(row.get("duration_ms") or 0.0)
                    calls = int(row.get("calls") or row.get("samples") or 1)
                    hot = max(0.0, duration_ms) * max(1, calls) / 1000.0
                _add(rel, hotness=hot, samples=row.get("samples") or row.get("calls") or 1, source=key)

    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            _add(
                row.get("file_path") or row.get("path") or row.get("file"),
                hotness=row.get("hotness") or row.get("duration_ms"),
                samples=row.get("samples") or row.get("calls") or 1,
                source="list",
            )

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
    structural = db.list_structural_duplicates(
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
    structural_hits = _count_file_hits_from_groups(structural)
    near_hits = _count_file_hits_from_pairs(near_pairs)

    payload = {
        "generated_at": _utc_now(),
        "exact_duplicate_groups": exact,
        "normalized_duplicate_groups": normalized,
        "structural_duplicate_groups": structural,
        "near_duplicate_pairs": near_pairs,
        "exact_hits_by_file": exact_hits,
        "normalized_hits_by_file": normalized_hits,
        "structural_hits_by_file": structural_hits,
        "near_hits_by_file": near_hits,
        "summary": {
            "exact_group_count": len(exact),
            "normalized_group_count": len(normalized),
            "structural_group_count": len(structural),
            "near_pair_count": len(near_pairs),
            "files_with_exact": len(exact_hits),
            "files_with_normalized": len(normalized_hits),
            "files_with_structural": len(structural_hits),
            "files_with_near": len(near_hits),
        },
    }

    (output_dir / "duplication_index.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_dependency_graph(
    db: CodeLibraryDB,
    output_dir: Path,
    *,
    rel_types: Optional[list[str]] = None,
    limit_edges: int = 20000,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = db.module_dependency_graph(rel_types=rel_types, limit_edges=limit_edges)
    payload["generated_at"] = _utc_now()
    (output_dir / "dependency_graph.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _classify_file(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    unit_count = int(metrics.get("unit_count") or 0)
    duplicate_pressure = float(metrics.get("duplicate_pressure") or 0.0)
    avg_complexity = float(metrics.get("avg_complexity") or 0.0)
    max_complexity = float(metrics.get("max_complexity") or 0.0)
    callable_units = int(metrics.get("callable_units") or 0)
    is_test = bool(metrics.get("is_test"))
    profile_hotness = float(metrics.get("profile_hotness") or 0.0)
    profile_samples = int(metrics.get("profile_samples") or 0)

    if unit_count < TRIAGE_THRESHOLDS["unit_count_min"]:
        reasons.append("No indexed code units detected for file")
        return {
            "label": "quarantine",
            "rule_id": "RULE_Q0_NO_INDEXED_UNITS",
            "confidence": 0.97,
            "reasons": reasons,
        }

    if is_test:
        reasons.append("Test file retained for behavioral proof and regression gates")
        return {
            "label": "keep",
            "rule_id": "RULE_K1_TEST_ASSET",
            "confidence": 0.98,
            "reasons": reasons,
        }

    if profile_hotness >= TRIAGE_THRESHOLDS["hot_path_keep_threshold"]:
        reasons.append(
            f"Runtime profile marks file as high-impact hot path (hotness={profile_hotness:.2f}, samples={profile_samples})"
        )
        if (
            max_complexity >= TRIAGE_THRESHOLDS["refactor_max_complexity"]
            or avg_complexity >= TRIAGE_THRESHOLDS["refactor_avg_complexity"]
        ):
            reasons.append("Preserve behavior and prioritize hotspot refactor hardening")
            return {
                "label": "refactor",
                "rule_id": "RULE_H1_HOT_PATH_REFACTOR",
                "confidence": 0.95,
                "reasons": reasons,
            }
        reasons.append("Preserve hot path stability during archive reduction")
        return {
            "label": "keep",
            "rule_id": "RULE_H2_HOT_PATH_KEEP",
            "confidence": 0.93,
            "reasons": reasons,
        }

    if (
        profile_hotness >= TRIAGE_THRESHOLDS["hot_path_preserve_threshold"]
        and duplicate_pressure >= TRIAGE_THRESHOLDS["delete_candidate_duplicate_pressure"]
        and unit_count <= TRIAGE_THRESHOLDS["delete_candidate_unit_count_max"]
    ):
        reasons.append(
            f"Hot-path profile signal ({profile_hotness:.2f}) overrides delete-candidate duplicate pressure ({duplicate_pressure:.2f})"
        )
        reasons.append("Extract to canonical module; avoid deleting runtime-sensitive file")
        return {
            "label": "extract",
            "rule_id": "RULE_H3_HOT_PATH_DELETE_OVERRIDE",
            "confidence": 0.9,
            "reasons": reasons,
        }

    if (
        duplicate_pressure >= TRIAGE_THRESHOLDS["delete_candidate_duplicate_pressure"]
        and unit_count <= TRIAGE_THRESHOLDS["delete_candidate_unit_count_max"]
    ):
        reasons.append(f"High duplication pressure ({duplicate_pressure:.2f}) with small unit surface")
        reasons.append("Candidate for deletion after extraction of unique capability")
        confidence = min(0.99, 0.80 + (duplicate_pressure / 10.0))
        return {
            "label": "delete_candidate",
            "rule_id": "RULE_D1_DUPLICATE_SMALL_SURFACE",
            "confidence": round(confidence, 4),
            "reasons": reasons,
        }

    if duplicate_pressure >= TRIAGE_THRESHOLDS["extract_duplicate_pressure"]:
        reasons.append(f"Elevated duplication pressure ({duplicate_pressure:.2f})")
        reasons.append("Retain behavior but extract into canonical reusable module")
        confidence = min(0.96, 0.65 + (duplicate_pressure / 6.0))
        return {
            "label": "extract",
            "rule_id": "RULE_E1_DUPLICATION_EXTRACTION",
            "confidence": round(confidence, 4),
            "reasons": reasons,
        }

    if (
        max_complexity >= TRIAGE_THRESHOLDS["refactor_max_complexity"]
        or avg_complexity >= TRIAGE_THRESHOLDS["refactor_avg_complexity"]
    ):
        reasons.append(f"Complexity hotspot detected (avg={avg_complexity:.2f}, max={max_complexity:.2f})")
        reasons.append("Refactor for modularity and testability")
        overload = max(
            max_complexity - TRIAGE_THRESHOLDS["refactor_max_complexity"],
            avg_complexity - TRIAGE_THRESHOLDS["refactor_avg_complexity"],
            0.0,
        )
        confidence = min(0.97, 0.72 + (overload / 20.0))
        return {
            "label": "refactor",
            "rule_id": "RULE_R1_COMPLEXITY_HOTSPOT",
            "confidence": round(confidence, 4),
            "reasons": reasons,
        }

    if callable_units >= TRIAGE_THRESHOLDS["keep_callable_units"]:
        reasons.append("Contains multiple callable symbols and low duplication")
        return {
            "label": "keep",
            "rule_id": "RULE_K2_CALLABLE_LOW_DUP",
            "confidence": 0.88,
            "reasons": reasons,
        }

    reasons.append("Insufficient evidence for direct keep/delete; hold for manual review")
    return {
        "label": "quarantine",
        "rule_id": "RULE_Q1_INSUFFICIENT_EVIDENCE",
        "confidence": 0.61,
        "reasons": reasons,
    }


def build_triage_report(
    db: CodeLibraryDB,
    repo_index: dict[str, Any],
    duplication_index: dict[str, Any],
    output_dir: Path,
    *,
    profile_hotspots: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_metrics = db.file_metrics(limit=max(1000, int(repo_index.get("files_total") or 0) * 4))
    metric_by_path = {str(rec.get("file_path")): rec for rec in file_metrics}

    exact_hits = {str(k): int(v) for k, v in (duplication_index.get("exact_hits_by_file") or {}).items()}
    normalized_hits = {str(k): int(v) for k, v in (duplication_index.get("normalized_hits_by_file") or {}).items()}
    structural_hits = {str(k): int(v) for k, v in (duplication_index.get("structural_hits_by_file") or {}).items()}
    near_hits = {str(k): int(v) for k, v in (duplication_index.get("near_hits_by_file") or {}).items()}

    normalized_hotspots = dict(profile_hotspots or {})

    entries: list[dict[str, Any]] = []
    audit_entries: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}

    for item in repo_index.get("entries", []):
        rel_path = str(item.get("path") or "")
        if not rel_path:
            continue

        dbm = metric_by_path.get(rel_path, {})
        unit_count = int(dbm.get("unit_count") or 0)

        profile = normalized_hotspots.get(rel_path) or normalized_hotspots.get(_normalize_profile_path(rel_path))

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
            "structural_duplicate_hits": structural_hits.get(rel_path, 0),
            "near_duplicate_hits": near_hits.get(rel_path, 0),
            "profile_hotness": float((profile or {}).get("hotness") or 0.0),
            "profile_samples": int((profile or {}).get("samples") or 0),
            "profile_source": str((profile or {}).get("source") or ""),
        }
        metrics["duplicate_pressure"] = (
            metrics["exact_duplicate_hits"]
            + metrics["normalized_duplicate_hits"]
            + metrics["structural_duplicate_hits"]
            + metrics["near_duplicate_hits"]
        ) / max(1, unit_count)

        decision = _classify_file(metrics)
        label = str(decision["label"])
        reasons = list(decision.get("reasons") or [])
        rule_id = str(decision.get("rule_id") or "RULE_UNKNOWN")
        confidence = float(decision.get("confidence") or 0.0)
        label_counts[label] = label_counts.get(label, 0) + 1

        entry = {
            "file_path": rel_path,
            "label": label,
            "rule_id": rule_id,
            "confidence": round(confidence, 4),
            "reasons": reasons,
            "metrics": metrics,
        }
        entries.append(entry)
        audit_entries.append(
            {
                "file_path": rel_path,
                "rule_id": rule_id,
                "label": label,
                "confidence": round(confidence, 4),
                "metrics": metrics,
                "reasons": reasons,
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
                "rule_id",
                "confidence",
                "language",
                "category",
                "unit_count",
                "duplicate_pressure",
                "avg_complexity",
                "max_complexity",
                "exact_duplicate_hits",
                "normalized_duplicate_hits",
                "structural_duplicate_hits",
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
                    rec.get("rule_id"),
                    rec.get("confidence"),
                    metrics.get("language"),
                    metrics.get("category"),
                    metrics.get("unit_count"),
                    round(float(metrics.get("duplicate_pressure") or 0.0), 4),
                    round(float(metrics.get("avg_complexity") or 0.0), 4),
                    round(float(metrics.get("max_complexity") or 0.0), 4),
                    metrics.get("exact_duplicate_hits"),
                    metrics.get("normalized_duplicate_hits"),
                    metrics.get("structural_duplicate_hits"),
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
            f"- `{rec['file_path']}` | label=`{rec['label']}` | rule=`{rec.get('rule_id')}` | confidence={rec.get('confidence', 0.0):.2f} | duplicate_pressure={m.get('duplicate_pressure', 0.0):.2f} | units={m.get('unit_count', 0)}"
        )

    if entries:
        avg_conf = sum(float(rec.get("confidence") or 0.0) for rec in entries) / len(entries)
    else:
        avg_conf = 0.0
    report_lines.append("")
    report_lines.append(f"Average confidence: {avg_conf:.4f}")

    if normalized_hotspots:
        report_lines.append("")
        report_lines.append("## Runtime Hot Paths")
        report_lines.append("")
        for rel, info in sorted(
            normalized_hotspots.items(),
            key=lambda item: float((item[1] or {}).get("hotness") or 0.0),
            reverse=True,
        )[:25]:
            report_lines.append(
                f"- `{rel}` | hotness={float((info or {}).get('hotness') or 0.0):.3f} | samples={int((info or {}).get('samples') or 0)}"
            )

    triage_audit = {
        "generated_at": _utc_now(),
        "ruleset_version": TRIAGE_RULESET_VERSION,
        "thresholds": TRIAGE_THRESHOLDS,
        "decisions": audit_entries,
    }
    (output_dir / "triage_audit.json").write_text(
        json.dumps(triage_audit, indent=2) + "\n",
        encoding="utf-8",
    )

    (output_dir / "triage_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    payload = {
        "generated_at": _utc_now(),
        "ruleset_version": TRIAGE_RULESET_VERSION,
        "entries": entries,
        "label_counts": label_counts,
        "profile_hotspots_count": len(normalized_hotspots),
        "csv_path": str(csv_path),
        "report_path": str(output_dir / "triage_report.md"),
        "triage_audit_path": str(output_dir / "triage_audit.json"),
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
    include_paths: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    progress_every: int = 200,
    sync_knowledge_path: Optional[Path] = None,
    sync_memory_path: Optional[Path] = None,
    graphrag_output_dir: Optional[Path] = None,
    graph_export_limit: int = 20000,
    integration_policy: str = "effective_run",
    strict_validation: bool = True,
    write_drift_report: bool = True,
    write_history_snapshot: bool = True,
    previous_snapshot_path: Optional[Path] = None,
    drift_warn_pct: float = 30.0,
    drift_min_abs_delta: float = 1.0,
    profile_trace_path: Optional[Path] = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = runner.ingest_path(
        Path(root_path),
        mode=mode,
        extensions=extensions,
        include_paths=include_paths,
        exclude_patterns=list(exclude_patterns) if exclude_patterns else None,
        max_files=max_files,
        progress_every=max(1, int(progress_every)),
    )

    repo_index = build_repo_index(
        root_path=Path(root_path),
        output_dir=output_dir,
        extensions=extensions,
        exclude_patterns=exclude_patterns,
        include_paths=include_paths,
        max_files=max_files,
    )
    duplication = build_duplication_index(db=db, output_dir=output_dir)
    dependency_graph = build_dependency_graph(db=db, output_dir=output_dir, limit_edges=graph_export_limit)
    profile_hotspots = load_profile_hotspots(profile_trace_path, root_path=Path(root_path))
    triage = build_triage_report(
        db=db,
        repo_index=repo_index,
        duplication_index=duplication,
        output_dir=output_dir,
        profile_hotspots=profile_hotspots,
    )

    normalized_policy = str(integration_policy).strip().lower() or "effective_run"
    if normalized_policy not in {"run", "effective_run", "global"}:
        raise ValueError("integration_policy must be one of: run, effective_run, global")

    integration_run_id: str | None
    if normalized_policy == "global":
        integration_run_id = None
    elif normalized_policy == "run":
        integration_run_id = str(stats.run_id)
    else:
        integration_run_id = str(stats.run_id)
        if int(stats.units_created) <= 0:
            effective = db.latest_effective_run_for_root(str(root_path), mode=mode)
            if effective and str(effective.get("run_id") or ""):
                integration_run_id = str(effective["run_id"])

    knowledge_sync = None
    if sync_knowledge_path is not None:
        knowledge_sync = sync_units_to_knowledge_forge(
            db=db,
            kb_path=Path(sync_knowledge_path),
            limit=max(1, int(graph_export_limit)),
            min_token_count=5,
            run_id=integration_run_id,
            include_node_links=True,
            node_links_limit=100,
        )

    memory_sync = None
    if sync_memory_path is not None:
        memory_sync = sync_units_to_memory_forge(
            db=db,
            memory_path=Path(sync_memory_path),
            limit=max(1, int(graph_export_limit)),
            min_token_count=8,
            run_id=integration_run_id,
            include_memory_links=True,
            memory_links_limit=100,
        )

    graphrag_export = None
    if graphrag_output_dir is not None:
        graphrag_export = export_units_for_graphrag(
            db=db,
            output_dir=Path(graphrag_output_dir),
            limit=max(1, int(graph_export_limit)),
            min_token_count=5,
            run_id=integration_run_id,
        )

    summary: dict[str, Any] = {
        "generated_at": _utc_now(),
        "root_path": str(Path(root_path).resolve()),
        "output_dir": str(output_dir),
        "ingestion_stats": asdict(stats),
        "repo_index_path": str(output_dir / "repo_index.json"),
        "duplication_index_path": str(output_dir / "duplication_index.json"),
        "dependency_graph_path": str(output_dir / "dependency_graph.json"),
        "triage_json_path": str(output_dir / "triage.json"),
        "triage_audit_path": str(output_dir / "triage_audit.json"),
        "triage_report_path": str(output_dir / "triage_report.md"),
        "profile_trace_path": str(Path(profile_trace_path).resolve()) if profile_trace_path else None,
        "profile_hotspots_count": int(triage.get("profile_hotspots_count") or 0),
        "knowledge_sync": knowledge_sync,
        "memory_sync": memory_sync,
        "graphrag_export": graphrag_export,
        "relationship_counts": db.relationship_counts(),
        "dependency_graph_summary": dependency_graph.get("summary", {}),
        "integration_policy": normalized_policy,
        "integration_run_id": integration_run_id,
        "provenance_path": None,
        "provenance_registry_path": None,
    }
    summary_path = output_dir / "archive_digester_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    provenance = write_provenance_links(
        output_path=output_dir / "provenance_links.json",
        stage="archive_digester",
        root_path=Path(root_path),
        source_run_id=str(stats.run_id),
        integration_policy=normalized_policy,
        integration_run_id=integration_run_id,
        artifact_paths=[
            ("repo_index", output_dir / "repo_index.json"),
            ("duplication_index", output_dir / "duplication_index.json"),
            ("dependency_graph", output_dir / "dependency_graph.json"),
            ("triage", output_dir / "triage.json"),
            ("triage_audit", output_dir / "triage_audit.json"),
            ("archive_summary", summary_path),
        ],
        knowledge_sync=knowledge_sync,
        memory_sync=memory_sync,
        graphrag_export=graphrag_export,
        extra={
            "mode": mode,
            "graph_export_limit": int(graph_export_limit),
            "files_processed": int(stats.files_processed),
            "units_created": int(stats.units_created),
        },
    )
    summary["provenance_path"] = provenance.get("path")

    validation = validate_output_dir(output_dir)
    summary["validation"] = validation

    if write_drift_report:
        drift = build_drift_report_from_output(
            output_dir=output_dir,
            previous_snapshot_path=Path(previous_snapshot_path).resolve() if previous_snapshot_path else None,
            history_dir=output_dir / "history",
            write_history=bool(write_history_snapshot),
            run_id=str(stats.run_id),
            warn_pct=float(drift_warn_pct),
            min_abs_delta=float(drift_min_abs_delta),
        )
        summary["drift"] = drift

    benchmark_payload = load_latest_benchmark_for_root(
        root_path=Path(root_path),
        search_roots=[output_dir, output_dir.parent, Path(root_path), Path(root_path) / "reports"],
    )
    registry = write_provenance_registry(
        output_path=output_dir / "provenance_registry.json",
        provenance_payload=provenance,
        stage_summary_payload=summary,
        drift_payload=summary.get("drift") if isinstance(summary.get("drift"), dict) else None,
        benchmark_payload=benchmark_payload,
    )
    summary["provenance_registry_path"] = registry.get("path")

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if strict_validation and not bool(validation.get("pass")):
        raise RuntimeError(f"Artifact validation failed: {validation.get('errors')}")
    return summary


def build_archive_reduction_plan(
    output_dir: Path,
    *,
    max_delete_candidates: int = 400,
    max_extract_candidates: int = 400,
    max_refactor_candidates: int = 400,
    max_quarantine_candidates: int = 200,
) -> dict[str, Any]:
    """
    Build candidate archive reduction plan artifacts from triage outputs.

    Outputs:
    - archive_reduction_plan.json
    - archive_reduction_plan.md
    """
    output_dir = Path(output_dir).resolve()
    triage_path = output_dir / "triage.json"
    if not triage_path.exists():
        raise FileNotFoundError(f"missing triage artifact: {triage_path}")

    triage = json.loads(triage_path.read_text(encoding="utf-8"))
    entries = list(triage.get("entries") or [])

    def _sorted_by_dup_then_conf(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda rec: (
                -float((rec.get("metrics") or {}).get("duplicate_pressure") or 0.0),
                -float(rec.get("confidence") or 0.0),
                str(rec.get("file_path") or ""),
            ),
        )

    delete_candidates = _sorted_by_dup_then_conf([rec for rec in entries if rec.get("label") == "delete_candidate"])[
        : max(1, int(max_delete_candidates))
    ]
    extract_candidates = _sorted_by_dup_then_conf([rec for rec in entries if rec.get("label") == "extract"])[
        : max(1, int(max_extract_candidates))
    ]
    refactor_candidates = sorted(
        [rec for rec in entries if rec.get("label") == "refactor"],
        key=lambda rec: (
            -float((rec.get("metrics") or {}).get("max_complexity") or 0.0),
            -float(rec.get("confidence") or 0.0),
            str(rec.get("file_path") or ""),
        ),
    )[: max(1, int(max_refactor_candidates))]
    quarantine_candidates = sorted(
        [rec for rec in entries if rec.get("label") == "quarantine"],
        key=lambda rec: (-float(rec.get("confidence") or 0.0), str(rec.get("file_path") or "")),
    )[: max(1, int(max_quarantine_candidates))]

    payload = {
        "generated_at": _utc_now(),
        "output_dir": str(output_dir),
        "source_triage_path": str(triage_path),
        "ruleset_version": triage.get("ruleset_version"),
        "counts": {
            "entries_total": len(entries),
            "delete_candidates": len(delete_candidates),
            "extract_candidates": len(extract_candidates),
            "refactor_candidates": len(refactor_candidates),
            "quarantine_candidates": len(quarantine_candidates),
        },
        "delete_candidates": delete_candidates,
        "extract_candidates": extract_candidates,
        "refactor_candidates": refactor_candidates,
        "quarantine_candidates": quarantine_candidates,
    }

    json_path = output_dir / "archive_reduction_plan.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Archive Reduction Plan",
        "",
        f"Generated: {payload['generated_at']}",
        f"Source triage: `{triage_path}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["counts"].items():
        lines.append(f"- `{key}`: {value}")

    def _section(title: str, rows: list[dict[str, Any]], score_key: str, metric_key: str) -> None:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("- (none)")
            return
        for rec in rows:
            metrics = rec.get("metrics") or {}
            lines.append(
                f"- `{rec.get('file_path')}` | label=`{rec.get('label')}` | {score_key}={float(rec.get('confidence') or 0.0):.2f} | {metric_key}={float(metrics.get(metric_key) or 0.0):.2f} | rule=`{rec.get('rule_id')}`"
            )

    _section("Delete Candidates", delete_candidates, "confidence", "duplicate_pressure")
    _section("Extract Candidates", extract_candidates, "confidence", "duplicate_pressure")
    _section("Refactor Candidates", refactor_candidates, "confidence", "max_complexity")
    _section("Quarantine Candidates", quarantine_candidates, "confidence", "duplicate_pressure")

    md_path = output_dir / "archive_reduction_plan.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    return payload


def build_triage_dashboard(
    output_dir: Path,
    *,
    dashboard_path: Optional[Path] = None,
    max_rows: int = 200,
) -> dict[str, Any]:
    """
    Generate a static HTML dashboard summarizing triage and duplication artifacts.
    """
    output_dir = Path(output_dir).resolve()
    triage_path = output_dir / "triage.json"
    duplication_path = output_dir / "duplication_index.json"
    if not triage_path.exists():
        raise FileNotFoundError(f"missing triage artifact: {triage_path}")
    if not duplication_path.exists():
        raise FileNotFoundError(f"missing duplication artifact: {duplication_path}")

    triage = json.loads(triage_path.read_text(encoding="utf-8"))
    duplication = json.loads(duplication_path.read_text(encoding="utf-8"))
    entries = list(triage.get("entries") or [])
    labels = dict(triage.get("label_counts") or {})

    top_dup = sorted(
        entries,
        key=lambda rec: float((rec.get("metrics") or {}).get("duplicate_pressure") or 0.0),
        reverse=True,
    )[: max(1, int(max_rows))]
    top_complexity = sorted(
        entries,
        key=lambda rec: float((rec.get("metrics") or {}).get("max_complexity") or 0.0),
        reverse=True,
    )[: max(1, int(max_rows))]

    exact_groups = list(duplication.get("exact_duplicate_groups") or [])
    normalized_groups = list(duplication.get("normalized_duplicate_groups") or [])
    structural_groups = list(duplication.get("structural_duplicate_groups") or [])
    near_pairs = list(duplication.get("near_duplicate_pairs") or [])

    dashboard_payload = {
        "generated_at": _utc_now(),
        "triage_path": str(triage_path),
        "duplication_path": str(duplication_path),
        "labels": labels,
        "summary": {
            "entries_total": len(entries),
            "exact_group_count": len(exact_groups),
            "normalized_group_count": len(normalized_groups),
            "structural_group_count": len(structural_groups),
            "near_pair_count": len(near_pairs),
        },
        "top_duplicate_pressure": [
            {
                "file_path": rec.get("file_path"),
                "label": rec.get("label"),
                "rule_id": rec.get("rule_id"),
                "confidence": float(rec.get("confidence") or 0.0),
                "duplicate_pressure": float(((rec.get("metrics") or {}).get("duplicate_pressure") or 0.0)),
            }
            for rec in top_dup
        ],
        "top_complexity": [
            {
                "file_path": rec.get("file_path"),
                "label": rec.get("label"),
                "rule_id": rec.get("rule_id"),
                "confidence": float(rec.get("confidence") or 0.0),
                "max_complexity": float(((rec.get("metrics") or {}).get("max_complexity") or 0.0)),
            }
            for rec in top_complexity
        ],
    }

    html_data = json.dumps(dashboard_payload, ensure_ascii=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Code Forge Dashboard</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #111827;
      --line: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #22d3ee;
      --good: #34d399;
      --warn: #f59e0b;
      --bad: #f87171;
    }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background: linear-gradient(145deg, var(--bg), #020617); color: var(--text); }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin: 16px 0; }}
    .card {{ background: color-mix(in srgb, var(--card) 85%, transparent); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }}
    .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .v {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .bar-wrap {{ margin-top: 12px; }}
    .bar-row {{ display: grid; grid-template-columns: 160px 1fr 50px; gap: 8px; align-items: center; margin: 4px 0; font-size: 13px; }}
    .bar {{ height: 10px; border-radius: 999px; background: #111827; border: 1px solid var(--line); overflow: hidden; }}
    .bar > span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--accent), #3b82f6); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 7px 6px; }}
    th {{ color: var(--muted); font-weight: 600; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Code Forge Dashboard</h1>
    <div class="muted" id="meta"></div>
    <div class="grid" id="summary"></div>
    <div class="card">
      <div class="k">Triage Labels</div>
      <div class="bar-wrap" id="labels"></div>
    </div>
    <div class="card">
      <div class="k">Top Duplicate Pressure</div>
      <table><thead><tr><th>File</th><th>Label</th><th>Rule</th><th>Pressure</th><th>Conf</th></tr></thead><tbody id="dup"></tbody></table>
    </div>
    <div class="card">
      <div class="k">Top Complexity</div>
      <table><thead><tr><th>File</th><th>Label</th><th>Rule</th><th>Max Complexity</th><th>Conf</th></tr></thead><tbody id="complexity"></tbody></table>
    </div>
  </div>
  <script>
    const data = {html_data};
    const meta = document.getElementById("meta");
    meta.textContent = "Generated " + data.generated_at + " | triage=" + data.triage_path + " | duplication=" + data.duplication_path;

    const summary = document.getElementById("summary");
    const summaryItems = [
      ["Entries", data.summary.entries_total],
      ["Exact groups", data.summary.exact_group_count],
      ["Normalized groups", data.summary.normalized_group_count],
      ["Structural groups", data.summary.structural_group_count],
      ["Near pairs", data.summary.near_pair_count],
    ];
    summary.innerHTML = summaryItems.map(([k, v]) => `<div class="card"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join("");

    const labelRoot = document.getElementById("labels");
    const entries = Object.entries(data.labels || {{}}).sort((a, b) => b[1] - a[1]);
    const max = Math.max(...entries.map(([, v]) => v), 1);
    labelRoot.innerHTML = entries.map(([label, value]) => {{
      const pct = Math.max(2, Math.round((value / max) * 100));
      return `<div class="bar-row"><div>${{label}}</div><div class="bar"><span style="width:${{pct}}%"></span></div><div>${{value}}</div></div>`;
    }}).join("");

    function rows(rows, key, targetId) {{
      const target = document.getElementById(targetId);
      target.innerHTML = rows.slice(0, 80).map((r) => `<tr><td>${{r.file_path || ""}}</td><td>${{r.label || ""}}</td><td>${{r.rule_id || ""}}</td><td>${{Number(r[key] || 0).toFixed(2)}}</td><td>${{Number(r.confidence || 0).toFixed(2)}}</td></tr>`).join("");
    }}
    rows(data.top_duplicate_pressure || [], "duplicate_pressure", "dup");
    rows(data.top_complexity || [], "max_complexity", "complexity");
  </script>
</body>
</html>
"""
    target = Path(dashboard_path).resolve() if dashboard_path else (output_dir / "dashboard.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return {
        "generated_at": dashboard_payload["generated_at"],
        "dashboard_path": str(target),
        "entries_total": int(dashboard_payload["summary"]["entries_total"]),
        "labels": labels,
    }
