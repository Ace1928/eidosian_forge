from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_ROOT", Path(__file__).resolve().parents[1])).resolve()
for extra in (
    FORGE_ROOT / "code_forge" / "src",
    FORGE_ROOT / "lib",
):
    text = str(extra)
    if extra.exists() and text not in sys.path:
        sys.path.insert(0, text)

from code_forge.analyzer.generic_analyzer import GenericCodeAnalyzer
from code_forge.digester.pipeline import (  # type: ignore
    initialize_archive_ingestion_state,
    load_archive_ingestion_state,
)
from code_forge.library.db import CodeLibraryDB  # type: ignore

DEFAULT_ARCHIVE_ROOT = FORGE_ROOT / "archive_forge"
DEFAULT_OUTPUT_DIR = FORGE_ROOT / "data" / "code_forge" / "archive_ingestion" / "latest"
DEFAULT_REPORT_DIR = FORGE_ROOT / "reports" / "code_forge_archive_plan"
RUNTIME_DIR = FORGE_ROOT / "data" / "runtime"
STATUS_PATH = RUNTIME_DIR / "code_forge_archive_plan_status.json"
HISTORY_PATH = RUNTIME_DIR / "code_forge_archive_plan_history.jsonl"
DEFAULT_RETENTION_POLICY_PATH = DEFAULT_OUTPUT_DIR / "repo_retention_policy.json"

DEFAULT_EXCLUDE_PATTERNS = [
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _archive_extensions() -> list[str]:
    extra = {
        ".md",
        ".rst",
        ".txt",
        ".adoc",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".csv",
        ".tsv",
        ".xml",
        ".html",
        ".htm",
        ".pdf",
        ".ipynb",
    }
    return sorted(set(GenericCodeAnalyzer.supported_extensions()) | extra)


def _normalize_include_paths(include_paths: Iterable[str] | None) -> set[str]:
    return {
        str(Path(item)).replace("\\", "/").lstrip("./")
        for item in (include_paths or [])
        if str(item).strip()
    }


def _should_skip(path: Path, exclude_patterns: Iterable[str]) -> bool:
    full = str(path)
    parts = set(path.parts)
    for pattern in exclude_patterns:
        token = str(pattern or "").strip()
        if not token:
            continue
        if "/" in token:
            if token in full:
                return True
            continue
        if token in parts:
            return True
    return False


def _category_for_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    suffix = path.suffix.lower()
    if any(p in {"test", "tests", "spec", "specs"} for p in parts) or path.name.startswith("test_"):
        return "test"
    if suffix in {".md", ".rst", ".txt", ".adoc", ".pdf", ".html", ".htm", ".ipynb"}:
        return "doc"
    if suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".conf", ".csv", ".tsv", ".xml"}:
        return "config"
    if suffix in {".sh", ".bash", ".zsh", ".ps1"}:
        return "script"
    if suffix in {".sql"}:
        return "data"
    return "source"


def _repo_key_for_path(rel_path: str) -> str:
    norm = str(rel_path).replace("\\", "/").lstrip("./")
    if not norm:
        return "."
    head = norm.split("/", 1)[0].strip()
    return head or "."


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


def _emit_planner_progress(*, archive_root: Path, output_dir: Path, scanned_files: int, indexed_files: int, last_path: str | None = None) -> None:
    current = _load_json(STATUS_PATH) or {}
    payload = dict(current)
    payload.update({
        "contract": "eidos.code_forge_archive_plan.status.v1",
        "status": "running",
        "phase": "planning",
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "scanned_files": int(scanned_files),
        "indexed_files": int(indexed_files),
        "last_path": last_path,
        "updated_at": _now_iso(),
    })
    _write_json(STATUS_PATH, payload)


def _entry_signature(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(entry.get("bytes") or 0),
        int(entry.get("modified_ns") or 0),
        str(entry.get("extension") or ""),
        str(entry.get("language") or ""),
        str(entry.get("category") or ""),
    )


def _diff_repo_indices(current: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    prev_entries = {}
    if isinstance(previous, dict):
        for item in list(previous.get("entries") or []):
            if isinstance(item, dict) and item.get("path"):
                prev_entries[str(item.get("path"))] = item
    curr_entries = {}
    for item in list(current.get("entries") or []):
        if isinstance(item, dict) and item.get("path"):
            curr_entries[str(item.get("path"))] = item

    added = sorted(set(curr_entries) - set(prev_entries))
    removed = sorted(set(prev_entries) - set(curr_entries))
    modified = sorted(
        path
        for path in sorted(set(curr_entries) & set(prev_entries))
        if _entry_signature(curr_entries[path]) != _entry_signature(prev_entries[path])
    )
    changed = sorted(set(added) | set(modified))
    return {
        "added_paths": added,
        "modified_paths": modified,
        "removed_paths": removed,
        "changed_paths": changed,
        "added_count": len(added),
        "modified_count": len(modified),
        "removed_count": len(removed),
        "changed_count": len(changed),
    }


def _fast_repo_index(
    root_path: Path,
    output_dir: Path,
    *,
    extensions: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    include_paths: Iterable[str] | None = None,
    max_files: int | None = None,
) -> dict[str, Any]:
    root_path = Path(root_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extension_set = {e.lower() for e in (extensions or _archive_extensions())}
    include_set = _normalize_include_paths(include_paths)
    exclude = list(exclude_patterns or DEFAULT_EXCLUDE_PATTERNS)

    entries: list[dict[str, Any]] = []
    by_language: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_extension: Counter[str] = Counter()
    by_repo: Counter[str] = Counter()

    scanned = 0
    accepted = 0

    def _handle_file(file_path: Path) -> bool:
        nonlocal scanned, accepted
        scanned += 1
        if _should_skip(file_path, exclude):
            if scanned % 5000 == 0:
                _emit_planner_progress(archive_root=root_path, output_dir=output_dir, scanned_files=scanned, indexed_files=accepted, last_path=str(file_path))
            return False
        suffix = file_path.suffix.lower()
        if suffix not in extension_set:
            if scanned % 5000 == 0:
                _emit_planner_progress(archive_root=root_path, output_dir=output_dir, scanned_files=scanned, indexed_files=accepted, last_path=str(file_path))
            return False
        try:
            stat = file_path.stat()
        except OSError:
            return False
        rel_path = file_path.relative_to(root_path).as_posix()
        language = GenericCodeAnalyzer.detect_language(file_path)
        category = _category_for_path(Path(rel_path))
        repo_key = _repo_key_for_path(rel_path)
        entries.append({
            "path": rel_path,
            "absolute_path": str(file_path),
            "repo_key": repo_key,
            "language": language,
            "extension": suffix,
            "category": category,
            "bytes": int(stat.st_size),
            "modified_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            "modified_ts": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
        by_language[language] += 1
        by_category[category] += 1
        by_extension[suffix] += 1
        by_repo[repo_key] += 1
        accepted += 1
        if scanned % 5000 == 0 or accepted % 2000 == 0:
            _emit_planner_progress(archive_root=root_path, output_dir=output_dir, scanned_files=scanned, indexed_files=accepted, last_path=rel_path)
        return True

    if include_set:
        for rel_path in sorted(include_set):
            file_path = (root_path / rel_path).resolve()
            if not file_path.is_file():
                continue
            _handle_file(file_path)
            if max_files is not None and accepted >= max(1, int(max_files)):
                break
    else:
        for dirpath, dirnames, filenames in os.walk(root_path):
            current_dir = Path(dirpath)
            dirnames[:] = [name for name in dirnames if not _should_skip(current_dir / name, exclude)]
            for filename in sorted(filenames):
                file_path = current_dir / filename
                _handle_file(file_path)
                if max_files is not None and accepted >= max(1, int(max_files)):
                    break
            if max_files is not None and accepted >= max(1, int(max_files)):
                break

    payload = {
        "generated_at": _now_iso(),
        "index_mode": "fast_metadata_v1",
        "root_path": str(root_path),
        "extensions": sorted(extension_set),
        "include_paths": sorted(include_set),
        "exclude_patterns": exclude,
        "files_total": len(entries),
        "scanned_total": int(scanned),
        "by_language": dict(sorted(by_language.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_extension": dict(sorted(by_extension.items())),
        "by_repo": dict(sorted(by_repo.items())),
        "entries": entries,
    }
    _write_json(output_dir / "repo_index.json", payload)
    _emit_planner_progress(archive_root=root_path, output_dir=output_dir, scanned_files=scanned, indexed_files=accepted, last_path=None)
    return payload


def _build_archive_ingestion_batches(
    repo_index: dict[str, Any],
    output_dir: Path,
    *,
    max_files_per_batch: int = 500,
    max_bytes_per_batch: int = 50_000_000,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    route_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for entry in list(repo_index.get("entries") or []):
        if not isinstance(entry, dict):
            continue
        route = _route_for_entry(entry)
        category = str(entry.get("category") or "unknown")
        repo_key = str(entry.get("repo_key") or ".")
        route_groups.setdefault((repo_key, route, category), []).append(entry)

    batches: list[dict[str, Any]] = []
    for (repo_key, route, category), entries in sorted(route_groups.items()):
        ordered = sorted(entries, key=lambda row: (str(row.get("extension") or ""), str(row.get("path") or "")))
        chunk: list[dict[str, Any]] = []
        chunk_bytes = 0
        chunk_no = 1

        def _flush() -> None:
            nonlocal chunk, chunk_bytes, chunk_no
            if not chunk:
                return
            batch_files = [str(item.get("path") or "") for item in chunk]
            batch_id = __import__("hashlib").sha256(
                f"{repo_key}|{route}|{category}|{chunk_no}|{'|'.join(batch_files)}".encode("utf-8")
            ).hexdigest()[:16]
            batches.append(
                {
                    "batch_id": batch_id,
                    "repo_key": repo_key,
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
                len(chunk) >= max(1, int(max_files_per_batch)) or (chunk_bytes + entry_bytes) > max(1, int(max_bytes_per_batch))
            ):
                _flush()
            chunk.append(entry)
            chunk_bytes += entry_bytes
        _flush()

    payload = {
        "generated_at": _now_iso(),
        "root_path": repo_index.get("root_path"),
        "files_total": int(repo_index.get("files_total") or 0),
        "batch_count": len(batches),
        "route_counts": {route: sum(1 for batch in batches if batch["route"] == route) for route in sorted({batch["route"] for batch in batches})},
        "repo_counts": {repo_key: sum(1 for batch in batches if batch["repo_key"] == repo_key) for repo_key in sorted({batch["repo_key"] for batch in batches})},
        "batches": batches,
    }
    _write_json(output_dir / "archive_ingestion_batches.json", payload)
    return payload


def _reconcile_state(
    batch_plan: dict[str, Any],
    previous_state: dict[str, Any] | None,
    *,
    changed_paths: set[str] | None = None,
) -> dict[str, Any]:
    changed_paths = changed_paths or set()
    state = {
        "generated_at": _now_iso(),
        "root_path": batch_plan.get("root_path"),
        "batch_count": 0,
        "completed_count": 0,
        "batches": {},
    }
    previous_batches = (previous_state or {}).get("batches") if isinstance(previous_state, dict) else {}
    if not isinstance(previous_batches, dict):
        previous_batches = {}

    for batch in list(batch_plan.get("batches") or []):
        if not isinstance(batch, dict):
            continue
        batch_id = str(batch.get("batch_id") or "")
        if not batch_id:
            continue
        prior = previous_batches.get(batch_id) if isinstance(previous_batches, dict) else None
        prior = prior if isinstance(prior, dict) else {}
        batch_paths = {str(path) for path in list(batch.get("paths") or []) if str(path).strip()}
        changed_in_batch = sorted(batch_paths & changed_paths)
        prior_status = str(prior.get("status") or "pending")
        if changed_in_batch or prior_status == "in_progress":
            status = "pending"
            last_error = None
            updated_at = _now_iso()
        else:
            status = prior_status
            last_error = prior.get("last_error")
            updated_at = prior.get("updated_at")
        state["batches"][batch_id] = {
            "repo_key": batch.get("repo_key"),
            "route": batch.get("route"),
            "category": batch.get("category"),
            "status": status,
            "attempts": int(prior.get("attempts") or 0),
            "last_error": last_error,
            "updated_at": updated_at,
            "changed_paths": changed_in_batch,
            "changed_path_count": len(changed_in_batch),
        }

    state["batch_count"] = len(state["batches"])
    state["completed_count"] = sum(1 for item in state["batches"].values() if str(item.get("status") or "") == "completed")
    return state


def _write_state(output_dir: Path, state: dict[str, Any]) -> Path:
    state_path = output_dir / "archive_ingestion_state.json"
    _write_json(state_path, state)
    return state_path


def _count_state_by_status(state: dict[str, Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    batches = state.get("batches") if isinstance(state, dict) else {}
    if isinstance(batches, dict):
        for item in batches.values():
            if isinstance(item, dict):
                counter[str(item.get("status") or "pending")] += 1
    return dict(sorted(counter.items()))


def _count_batches_by_route(batch_plan: dict[str, Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for batch in list(batch_plan.get("batches") or []):
        if isinstance(batch, dict):
            counter[str(batch.get("route") or "unknown")] += 1
    return dict(sorted(counter.items()))


def _sum_batch_bytes(batch_plan: dict[str, Any]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for batch in list(batch_plan.get("batches") or []):
        if isinstance(batch, dict):
            totals[str(batch.get("route") or "unknown")] += int(batch.get("total_bytes") or 0)
    return dict(sorted(totals.items()))


def _summarize_repos(repo_index: dict[str, Any], batch_plan: dict[str, Any], state: dict[str, Any], limit: int = 32) -> list[dict[str, Any]]:
    file_counts: Counter[str] = Counter()
    byte_counts: Counter[str] = Counter()
    for entry in list(repo_index.get("entries") or []):
        if not isinstance(entry, dict):
            continue
        repo_key = str(entry.get("repo_key") or ".")
        file_counts[repo_key] += 1
        byte_counts[repo_key] += int(entry.get("bytes") or 0)

    batch_counts: Counter[str] = Counter()
    completed_counts: Counter[str] = Counter()
    failed_counts: Counter[str] = Counter()
    for batch in list(batch_plan.get("batches") or []):
        if not isinstance(batch, dict):
            continue
        batch_id = str(batch.get("batch_id") or "")
        repo_key = str(batch.get("repo_key") or ".")
        batch_counts[repo_key] += 1
        item = ((state.get("batches") or {}).get(batch_id) or {}) if isinstance(state, dict) else {}
        status = str(item.get("status") or "pending")
        if status == "completed":
            completed_counts[repo_key] += 1
        if status == "failed":
            failed_counts[repo_key] += 1

    rows: list[dict[str, Any]] = []
    for repo_key in sorted(file_counts, key=lambda key: (int(byte_counts.get(key, 0)), key), reverse=True):
        rows.append(
            {
                "repo_key": repo_key,
                "file_count": int(file_counts.get(repo_key, 0)),
                "bytes": int(byte_counts.get(repo_key, 0)),
                "batch_count": int(batch_counts.get(repo_key, 0)),
                "completed_batches": int(completed_counts.get(repo_key, 0)),
                "failed_batches": int(failed_counts.get(repo_key, 0)),
            }
        )
    return rows[: max(1, int(limit))]


def _provenance_counts(output_dir: Path) -> dict[str, int]:
    link_count = 0
    registry_count = 0
    for path in output_dir.rglob("provenance_*.json"):
        if path.name == "provenance_links.json":
            link_count += 1
        elif path.name == "provenance_registry.json":
            registry_count += 1
    return {
        "provenance_link_count": link_count,
        "provenance_registry_count": registry_count,
    }


def _largest_assets(base: Path, repo_root: Path, limit: int = 12) -> list[dict[str, Any]]:
    if not base.exists():
        return []
    rows: list[dict[str, Any]] = []
    suffixes = {".bin", ".sqlite", ".sqlite-wal", ".sqlite-shm", ".graphml", ".gexf", ".npy", ".npz", ".hnsw"}
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in suffixes and not path.name.endswith((".sqlite-wal", ".sqlite-shm")):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        rows.append({"path": str(path.relative_to(repo_root.resolve())), "bytes": size})
    rows.sort(key=lambda row: int(row.get("bytes") or 0), reverse=True)
    return rows[: max(1, int(limit))]


def _lfs_patterns(repo_root: Path) -> list[str]:
    gitattributes = repo_root / ".gitattributes"
    if not gitattributes.exists():
        return []
    lines = []
    for line in gitattributes.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "filter=lfs" in stripped and "code_forge" in stripped:
            lines.append(stripped)
    return lines


def _code_library_summary(repo_root: Path) -> dict[str, Any]:
    db_path = repo_root / "data" / "code_forge" / "library.sqlite"
    if not db_path.exists():
        return {"db_path": str(db_path), "present": False}
    db = CodeLibraryDB(db_path)
    try:
        return {
            "db_path": str(db_path),
            "present": True,
            "unit_count": db.count_units(),
            "relationship_counts": db.relationship_counts(),
            "languages": db.count_units_by_language(),
            "unit_types": db.count_units_by_type(),
            "file_record_count": db.count_file_records(),
        }
    finally:
        try:
            db.close()
        except Exception:
            pass


def build_archive_plan_report(
    *,
    repo_root: Path,
    archive_root: Path,
    output_dir: Path,
    refresh: bool = False,
    max_files: int | None = None,
    max_files_per_batch: int = 500,
    max_bytes_per_batch: int = 50_000_000,
) -> dict[str, Any]:
    archive_root = archive_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_index_path = output_dir / "repo_index.json"
    batch_plan_path = output_dir / "archive_ingestion_batches.json"
    state_path = output_dir / "archive_ingestion_state.json"
    previous_repo_index = _load_json(repo_index_path) if repo_index_path.exists() else None

    repo_index = _fast_repo_index(
        archive_root,
        output_dir,
        extensions=_archive_extensions(),
        max_files=max_files,
    ) if refresh or not repo_index_path.exists() else (_load_json(repo_index_path) or _fast_repo_index(archive_root, output_dir, extensions=_archive_extensions(), max_files=max_files))

    previous_state = _load_json(state_path) if state_path.exists() else None
    batch_plan = _build_archive_ingestion_batches(
        repo_index,
        output_dir,
        max_files_per_batch=max(1, int(max_files_per_batch)),
        max_bytes_per_batch=max(1, int(max_bytes_per_batch)),
    ) if refresh or not batch_plan_path.exists() else (_load_json(batch_plan_path) or _build_archive_ingestion_batches(repo_index, output_dir, max_files_per_batch=max(1, int(max_files_per_batch)), max_bytes_per_batch=max(1, int(max_bytes_per_batch))))

    diff = _diff_repo_indices(repo_index, previous_repo_index)
    if refresh and previous_state:
        state = _reconcile_state(batch_plan, previous_state, changed_paths=set(diff["changed_paths"]))
        _write_state(output_dir, state)
    elif state_path.exists():
        state = load_archive_ingestion_state(output_dir)
    else:
        state = initialize_archive_ingestion_state(batch_plan, output_dir)

    status_counts = _count_state_by_status(state)
    route_counts = _count_batches_by_route(batch_plan)
    route_bytes = _sum_batch_bytes(batch_plan)
    prov_counts = _provenance_counts(output_dir)
    total_batches = int(batch_plan.get("batch_count") or len(batch_plan.get("batches") or []))
    completed_batches = int(state.get("completed_count") or 0)
    retirement_ready = (
        total_batches > 0
        and completed_batches == total_batches
        and status_counts.get("failed", 0) == 0
        and status_counts.get("in_progress", 0) == 0
        and prov_counts["provenance_link_count"] >= completed_batches
        and prov_counts["provenance_registry_count"] >= completed_batches
    )

    report = {
        "contract": "eidos.code_forge_archive_plan.v2",
        "generated_at": _now_iso(),
        "repo_root": str(repo_root),
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "repo_index_path": str(repo_index_path),
        "batch_plan_path": str(batch_plan_path),
        "state_path": str(state_path),
        "retention_policy_path": str(output_dir / "repo_retention_policy.json"),
        "retention_modes_supported": ["ingest_and_keep", "ingest_and_remove"],
        "archive_files_total": int(repo_index.get("files_total") or 0),
        "archive_bytes_total": sum(int(item.get("bytes") or 0) for item in list(repo_index.get("entries") or []) if isinstance(item, dict)),
        "repo_count": len(repo_index.get("by_repo") or {}),
        "by_language": repo_index.get("by_language") or {},
        "by_category": repo_index.get("by_category") or {},
        "batch_count": total_batches,
        "route_batch_counts": route_counts,
        "route_byte_totals": route_bytes,
        "state_status_counts": status_counts,
        "completed_count": completed_batches,
        "pending_count": status_counts.get("pending", 0),
        "failed_count": status_counts.get("failed", 0),
        "change_summary": diff,
        "repo_summaries": _summarize_repos(repo_index, batch_plan, state),
        "provenance": prov_counts,
        "retirement": {
            "ready": retirement_ready,
            "reason": "all batches completed with matching provenance artifacts" if retirement_ready else "archive still lacks completed reversible ingestion evidence",
        },
        "code_library": _code_library_summary(repo_root),
        "vector_graph_assets": _largest_assets(repo_root / "data" / "code_forge", repo_root),
        "git_lfs_patterns": _lfs_patterns(repo_root),
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    retirement = report.get("retirement") if isinstance(report.get("retirement"), dict) else {}
    change_summary = report.get("change_summary") if isinstance(report.get("change_summary"), dict) else {}
    lines = [
        "# Code Forge Archive Plan",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Archive root: `{report.get('archive_root')}`",
        f"- Output dir: `{report.get('output_dir')}`",
        f"- Archive files: `{report.get('archive_files_total')}`",
        f"- Archive bytes: `{report.get('archive_bytes_total')}`",
        f"- Repo count: `{report.get('repo_count')}`",
        f"- Batch count: `{report.get('batch_count')}`",
        f"- Completed batches: `{report.get('completed_count')}`",
        f"- Pending batches: `{report.get('pending_count')}`",
        f"- Failed batches: `{report.get('failed_count')}`",
        f"- Changed paths: `{change_summary.get('changed_count', 0)}`",
        f"- Added paths: `{change_summary.get('added_count', 0)}`",
        f"- Modified paths: `{change_summary.get('modified_count', 0)}`",
        f"- Removed paths: `{change_summary.get('removed_count', 0)}`",
        f"- Provenance links: `{((report.get('provenance') or {}).get('provenance_link_count'))}`",
        f"- Provenance registries: `{((report.get('provenance') or {}).get('provenance_registry_count'))}`",
        f"- Retirement ready: `{retirement.get('ready')}`",
        f"- Retirement note: `{retirement.get('reason')}`",
        "",
        "## Routes",
        "",
        "| Route | Batches | Bytes |",
        "| --- | ---: | ---: |",
    ]
    route_counts = report.get("route_batch_counts") if isinstance(report.get("route_batch_counts"), dict) else {}
    route_bytes = report.get("route_byte_totals") if isinstance(report.get("route_byte_totals"), dict) else {}
    for route in sorted(route_counts):
        lines.append(f"| {route} | {route_counts.get(route, 0)} | {route_bytes.get(route, 0)} |")
    lines.extend(["", "## Repo Summary", "", "| Repo | Files | Bytes | Batches | Completed | Failed |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    repo_rows = report.get("repo_summaries") if isinstance(report.get("repo_summaries"), list) else []
    if repo_rows:
        for row in repo_rows:
            if isinstance(row, dict):
                lines.append(f"| {row.get('repo_key')} | {row.get('file_count')} | {row.get('bytes')} | {row.get('batch_count')} | {row.get('completed_batches')} | {row.get('failed_batches')} |")
    else:
        lines.append("| none | 0 | 0 | 0 | 0 | 0 |")
    lines.extend(["", "## State", "", "| Status | Count |", "| --- | ---: |"])
    for status, count in sorted(((report.get("state_status_counts") or {}).items())):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Git LFS Patterns", ""])
    patterns = report.get("git_lfs_patterns") if isinstance(report.get("git_lfs_patterns"), list) else []
    if patterns:
        for pattern in patterns:
            lines.append(f"- `{pattern}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Largest Vector / Graph Assets", "", "| Path | Bytes |", "| --- | ---: |"])
    assets = report.get("vector_graph_assets") if isinstance(report.get("vector_graph_assets"), list) else []
    if assets:
        for row in assets:
            if isinstance(row, dict):
                lines.append(f"| {row.get('path')} | {row.get('bytes')} |")
    else:
        lines.append("| none | 0 |")
    return "\n".join(lines) + "\n"


def write_archive_plan_report(
    *,
    repo_root: Path,
    archive_root: Path,
    output_dir: Path,
    report_dir: Path,
    refresh: bool = False,
    max_files: int | None = None,
    max_files_per_batch: int = 500,
    max_bytes_per_batch: int = 50_000_000,
) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_archive_plan_report(
        repo_root=repo_root,
        archive_root=archive_root,
        output_dir=output_dir,
        refresh=refresh,
        max_files=max_files,
        max_files_per_batch=max_files_per_batch,
        max_bytes_per_batch=max_bytes_per_batch,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"code_forge_archive_plan_{stamp}.json"
    md_path = report_dir / f"code_forge_archive_plan_{stamp}.md"
    latest_json = report_dir / "latest.json"
    latest_md = report_dir / "latest.md"
    _write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "report": report,
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/update the full Code Forge archive ingestion plan and retirement-readiness report.")
    parser.add_argument("--repo-root", default=str(FORGE_ROOT))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-files-per-batch", type=int, default=500)
    parser.add_argument("--max-bytes-per-batch", type=int, default=50_000_000)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    archive_root = Path(args.archive_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_dir = Path(args.report_dir).resolve()

    running = {
        "contract": "eidos.code_forge_archive_plan.status.v1",
        "status": "running",
        "started_at": _now_iso(),
        "phase": "planning",
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "refresh": bool(args.refresh),
        "index_mode": "fast_metadata_v1",
    }
    _write_json(STATUS_PATH, running)
    _append_jsonl(HISTORY_PATH, running)
    try:
        result = write_archive_plan_report(
            repo_root=repo_root,
            archive_root=archive_root,
            output_dir=output_dir,
            report_dir=report_dir,
            refresh=bool(args.refresh),
            max_files=args.max_files,
            max_files_per_batch=max(1, int(args.max_files_per_batch)),
            max_bytes_per_batch=max(1, int(args.max_bytes_per_batch)),
        )
        completed = {
            "contract": "eidos.code_forge_archive_plan.status.v1",
            "status": "completed",
            "phase": "planning",
            "started_at": running["started_at"],
            "finished_at": _now_iso(),
            "archive_root": str(archive_root),
            "output_dir": str(output_dir),
            "report_dir": str(report_dir),
            "refresh": bool(args.refresh),
            "index_mode": "fast_metadata_v1",
            "json_path": result["json_path"],
            "markdown_path": result["markdown_path"],
            "latest_json": result["latest_json"],
            "latest_markdown": result["latest_markdown"],
            "archive_files_total": result["report"].get("archive_files_total"),
            "batch_count": result["report"].get("batch_count"),
            "completed_count": result["report"].get("completed_count"),
            "retirement_ready": ((result["report"].get("retirement") or {}).get("ready")),
        }
        _write_json(STATUS_PATH, completed)
        _append_jsonl(HISTORY_PATH, completed)
        print(json.dumps(result["report"], indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failed = {
            "contract": "eidos.code_forge_archive_plan.status.v1",
            "status": "error",
            "phase": "planning",
            "started_at": running["started_at"],
            "finished_at": _now_iso(),
            "archive_root": str(archive_root),
            "output_dir": str(output_dir),
            "report_dir": str(report_dir),
            "refresh": bool(args.refresh),
            "index_mode": "fast_metadata_v1",
            "error": str(exc),
        }
        _write_json(STATUS_PATH, failed)
        _append_jsonl(HISTORY_PATH, failed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
