from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_ROOT", Path(__file__).resolve().parents[1])).resolve()
for extra in (
    FORGE_ROOT / "scripts",
    FORGE_ROOT / "lib",
    FORGE_ROOT / "code_forge" / "src",
    FORGE_ROOT / "gis_forge" / "src",
    FORGE_ROOT / "crawl_forge" / "src",
    FORGE_ROOT / "knowledge_forge" / "src",
    FORGE_ROOT / "memory_forge" / "src",
    FORGE_ROOT / "word_forge" / "src",
    FORGE_ROOT / "doc_forge" / "src",
    FORGE_ROOT / "eidos_mcp" / "src",
    FORGE_ROOT / "file_forge" / "src",
):
    text = str(extra)
    if extra.exists() and text not in sys.path:
        sys.path.insert(0, text)

from code_forge.digester.pipeline import run_archive_ingestion_batches  # type: ignore
from code_forge.ingest.runner import IngestionRunner  # type: ignore
from code_forge.library.db import CodeLibraryDB  # type: ignore
from code_forge.reconstruct.pipeline import build_reconstruction_from_library, compare_tree_parity  # type: ignore
from code_forge_archive_plan import build_archive_plan_report, _repo_key_for_path  # type: ignore
from file_forge import FileForge, FileLibraryDB  # type: ignore

DEFAULT_ARCHIVE_ROOT = FORGE_ROOT / "archive_forge"
DEFAULT_OUTPUT_DIR = FORGE_ROOT / "data" / "code_forge" / "archive_ingestion" / "latest"
DEFAULT_REPORT_DIR = FORGE_ROOT / "reports" / "code_forge_archive_lifecycle"
RUNTIME_DIR = FORGE_ROOT / "data" / "runtime"
STATUS_PATH = RUNTIME_DIR / "code_forge_archive_lifecycle_status.json"
HISTORY_PATH = RUNTIME_DIR / "code_forge_archive_lifecycle_history.jsonl"
POLICY_PATH = DEFAULT_OUTPUT_DIR / "repo_retention_policy.json"
RETIREMENTS_DIR = DEFAULT_OUTPUT_DIR / "retirements"
RETIREMENTS_LATEST = RETIREMENTS_DIR / "latest.json"
RETIREMENTS_HISTORY = RETIREMENTS_DIR / "history.jsonl"


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


def _write_runtime_status(payload: dict[str, Any]) -> None:
    _write_json(STATUS_PATH, payload)
    _append_jsonl(HISTORY_PATH, payload)


def _default_policy() -> dict[str, Any]:
    return {
        "contract": "eidos.code_forge_retention_policy.v1",
        "generated_at": _now_iso(),
        "default_mode": "ingest_and_keep",
        "repos": {},
    }


def load_or_init_policy(path: Path) -> dict[str, Any]:
    policy = _load_json(path)
    if not isinstance(policy, dict):
        policy = _default_policy()
        _write_json(path, policy)
    repos = policy.get("repos")
    if not isinstance(repos, dict):
        policy["repos"] = {}
    if not policy.get("default_mode"):
        policy["default_mode"] = "ingest_and_keep"
    return policy


def set_repo_mode(policy_path: Path, repo_key: str, mode: str, reason: str | None = None) -> dict[str, Any]:
    if mode not in {"ingest_and_keep", "ingest_and_remove"}:
        raise ValueError(f"unsupported mode: {mode}")
    policy = load_or_init_policy(policy_path)
    repos = policy.setdefault("repos", {})
    assert isinstance(repos, dict)
    repos[str(repo_key)] = {
        "mode": mode,
        "updated_at": _now_iso(),
        "reason": str(reason or "").strip() or None,
    }
    policy["generated_at"] = _now_iso()
    _write_json(policy_path, policy)
    return policy


def _repo_mode(policy: dict[str, Any], repo_key: str) -> str:
    repos = policy.get("repos") if isinstance(policy, dict) else {}
    if isinstance(repos, dict):
        rec = repos.get(repo_key)
        if isinstance(rec, dict) and str(rec.get("mode") or "") in {"ingest_and_keep", "ingest_and_remove"}:
            return str(rec.get("mode"))
    return str(policy.get("default_mode") or "ingest_and_keep")


def _batch_provenance_state(output_dir: Path, batch_id: str) -> dict[str, bool]:
    batch_dir = output_dir / "batches" / batch_id
    return {
        "links": (batch_dir / "provenance_links.json").exists(),
        "registry": (batch_dir / "provenance_registry.json").exists(),
    }


def _retirement_manifests(output_dir: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    retirements_dir = output_dir / "retirements"
    for path in retirements_dir.glob("*/retirement_manifest.json"):
        payload = _load_json(path)
        if isinstance(payload, dict) and payload.get("repo_key"):
            manifests[str(payload.get("repo_key"))] = payload
    latest = _load_json(retirements_dir / "latest.json")
    if isinstance(latest, dict):
        for row in list(latest.get("retirements") or []):
            if isinstance(row, dict) and row.get("repo_key"):
                manifests.setdefault(str(row.get("repo_key")), row)
    return manifests


def _retirement_ready_for_row(row: dict[str, Any], *, effective_mode: str | None = None) -> bool:
    return not _retirement_blockers_for_row(row, effective_mode=effective_mode)


def _retirement_blockers_for_row(row: dict[str, Any], *, effective_mode: str | None = None) -> list[str]:
    batch_count = max(0, int(row.get("batch_count") or 0))
    mode = str(effective_mode or row.get("mode") or "ingest_and_keep")
    completed = int(row.get("completed_batches") or 0)
    failed = int(row.get("failed_batches") or 0)
    pending = int(row.get("pending_batches") or 0)
    provenance_links = int(row.get("provenance_links") or 0)
    provenance_registries = int(row.get("provenance_registries") or 0)
    file_records = int(row.get("file_record_count") or 0)
    file_count = int(row.get("file_count") or 0)
    source_tree_file_count = int(row.get("source_tree_file_count") or 0)
    source_tree_unindexed_count = int(row.get("source_tree_unindexed_count") or 0)
    source_tree_unindexed_reversible_count = int(row.get("source_tree_unindexed_reversible_count") or 0)
    reversible_file_count = int(row.get("reversible_file_count") or 0)
    blockers: list[str] = []
    if mode != "ingest_and_remove":
        blockers.append("repo mode is not ingest_and_remove")
    if bool(row.get("retired")):
        blockers.append("repo is already retired")
    if batch_count <= 0:
        blockers.append("no archive ingestion batches are planned")
    if completed < batch_count:
        blockers.append(f"completed_batches {completed}/{batch_count}")
    if failed > 0:
        blockers.append(f"failed_batches={failed}")
    if pending > 0:
        blockers.append(f"pending_batches={pending}")
    if provenance_links < batch_count:
        blockers.append(f"provenance_links {provenance_links}/{batch_count}")
    if provenance_registries < batch_count:
        blockers.append(f"provenance_registries {provenance_registries}/{batch_count}")
    if file_records < file_count:
        blockers.append(f"file_records {file_records}/{file_count}")
    if source_tree_file_count > 0 and reversible_file_count < source_tree_file_count:
        blockers.append(f"reversible_files {reversible_file_count}/{source_tree_file_count}")
    if source_tree_unindexed_count > 0 and source_tree_unindexed_reversible_count < source_tree_unindexed_count:
        blockers.append(f"unindexed_source_files {source_tree_unindexed_reversible_count}/{source_tree_unindexed_count}")
    return blockers


def _normalize_archive_entry_path(entry_path: str, *, archive_root: Path) -> str:
    path = Path(str(entry_path or ""))
    if not path.is_absolute():
        path = archive_root / path
    return str(path.resolve())


def _batch_captured_paths(output_dir: Path, batch_id: str, *, archive_root: Path) -> set[str]:
    batch_dir = output_dir / "batches" / batch_id
    captured: set[str] = set()
    repo_index_path = batch_dir / "repo_index.json"
    if repo_index_path.exists():
        payload = _load_json(repo_index_path) or {}
        for entry in list(payload.get("entries") or []):
            if isinstance(entry, dict) and entry.get("path"):
                captured.add(_normalize_archive_entry_path(str(entry.get("path")), archive_root=archive_root))
    for summary_name in ("document_batch_summary.json", "metadata_batch_summary.json"):
        summary_path = batch_dir / summary_name
        if not summary_path.exists():
            continue
        payload = _load_json(summary_path) or {}
        for result in list(payload.get("results") or []):
            if not isinstance(result, dict):
                continue
            rel_path = result.get("path")
            if rel_path:
                captured.add(_normalize_archive_entry_path(str(rel_path), archive_root=archive_root))
                continue
            ingest = result.get("ingest") or {}
            source = ingest.get("source") if isinstance(ingest, dict) else None
            if source:
                captured.add(str(Path(str(source)).resolve()))
    return captured


def _file_forge_db_path(repo_root: Path) -> Path:
    return (repo_root / "data" / "file_forge" / "library.sqlite").resolve()


def _repo_source_file_paths(source_root: Path) -> list[str]:
    if not source_root.exists():
        return []
    return [
        str(path.resolve())
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
    ]


def _run_file_forge_index(
    *,
    repo_root: Path,
    source_root: Path,
    repo_key: str,
    remove_after_ingest: bool = False,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.exists():
        return {
            "repo_key": repo_key,
            "status": "missing_source",
            "source_root": str(source_root),
            "indexed": 0,
            "skipped": 0,
            "removed": 0,
        }
    forge = FileForge(base_path=repo_root)
    result = forge.index_directory(
        source_root,
        db_path=_file_forge_db_path(repo_root),
        remove_after_ingest=bool(remove_after_ingest),
    )
    result["repo_key"] = repo_key
    result["source_root"] = str(source_root)
    return result


def _build_unified_reconstruction(
    *,
    repo_root: Path,
    code_db: CodeLibraryDB,
    file_db: FileLibraryDB,
    source_root: Path,
    output_dir: Path,
    strict: bool = True,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    code_manifest = build_reconstruction_from_library(
        db=code_db,
        source_root=source_root,
        output_dir=output_dir,
        strict=bool(strict),
    )
    forge = FileForge(base_path=repo_root)
    file_result = forge.restore_directory(
        source_root,
        target_root=output_dir,
        db_path=_file_forge_db_path(repo_root),
        overwrite=True,
    )
    manifest = {
        "generated_at": _now_iso(),
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "strict": bool(strict),
        "code_forge_manifest_path": code_manifest.get("manifest_path"),
        "code_forge_files_written": int(code_manifest.get("files_written") or 0),
        "code_forge_records_scanned": int(code_manifest.get("records_scanned") or 0),
        "file_forge_restored": int(file_result.get("restored") or 0),
        "file_forge_skipped_existing": int(file_result.get("skipped_existing") or 0),
        "file_forge_overwritten_existing": int(file_result.get("overwritten_existing") or 0),
        "file_forge_missing_records": int(file_result.get("missing_records") or 0),
        "file_forge_missing_blobs": int(file_result.get("missing_blobs") or 0),
        "files_written_total": int(code_manifest.get("files_written") or 0) + int(file_result.get("restored") or 0),
        "results": {
            "code_forge": code_manifest,
            "file_forge": file_result,
        },
    }
    manifest_path = output_dir.parent / f"{output_dir.name}_combined_reconstruction_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    if strict and (int(file_result.get("missing_records") or 0) > 0 or int(file_result.get("missing_blobs") or 0) > 0):
        raise RuntimeError(
            f"file forge reconstruction incomplete: missing_records={file_result.get('missing_records')} missing_blobs={file_result.get('missing_blobs')}"
        )
    return manifest


def build_repo_status_report(
    *,
    repo_root: Path,
    archive_root: Path,
    output_dir: Path,
    report_dir: Path,
    repo_keys: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    archive_root = archive_root.resolve()
    output_dir = output_dir.resolve()
    report_dir = report_dir.resolve()
    policy = load_or_init_policy(output_dir / "repo_retention_policy.json")
    repo_index = _load_json(output_dir / "repo_index.json") or {}
    batch_plan = _load_json(output_dir / "archive_ingestion_batches.json") or {}
    state = _load_json(output_dir / "archive_ingestion_state.json") or {}
    retirement_manifests = _retirement_manifests(output_dir)
    selected_repo_keys = {str(key) for key in (repo_keys or []) if str(key).strip()}

    db = CodeLibraryDB(repo_root / "data" / "code_forge" / "library.sqlite")
    file_db = FileLibraryDB(_file_forge_db_path(repo_root))
    try:
        entries_by_repo: dict[str, list[dict[str, Any]]] = {}
        for entry in list(repo_index.get("entries") or []):
            if isinstance(entry, dict):
                repo_key = str(entry.get("repo_key") or _repo_key_for_path(str(entry.get("path") or "")))
                entries_by_repo.setdefault(repo_key, []).append(entry)

        batches_by_repo: dict[str, list[dict[str, Any]]] = {}
        for batch in list(batch_plan.get("batches") or []):
            if isinstance(batch, dict):
                repo_key = str(batch.get("repo_key") or ".")
                batches_by_repo.setdefault(repo_key, []).append(batch)

        rows: list[dict[str, Any]] = []
        repo_names = sorted(selected_repo_keys or set(entries_by_repo))
        for repo_key in repo_names:
            entries = entries_by_repo.get(repo_key) or []
            batches = batches_by_repo.get(repo_key) or []
            source_root = (archive_root / repo_key).resolve()
            file_count = len(entries)
            byte_count = sum(int(entry.get("bytes") or 0) for entry in entries if isinstance(entry, dict))
            completed = 0
            failed = 0
            pending = 0
            prov_links = 0
            prov_registry = 0
            captured_file_paths: set[str] = set()
            for batch in batches:
                batch_id = str(batch.get("batch_id") or "")
                batch_state = ((state.get("batches") or {}).get(batch_id) or {}) if isinstance(state, dict) else {}
                status = str(batch_state.get("status") or "pending")
                if status == "completed":
                    completed += 1
                elif status == "failed":
                    failed += 1
                else:
                    pending += 1
                prov = _batch_provenance_state(output_dir, batch_id)
                prov_links += 1 if prov["links"] else 0
                prov_registry += 1 if prov["registry"] else 0
                captured_file_paths.update(_batch_captured_paths(output_dir, batch_id, archive_root=archive_root))

            mode = _repo_mode(policy, repo_key)
            tracked_file_paths = {
                str(Path(str(record.get("file_path") or "")).resolve())
                for record in db.iter_file_records(path_prefix=source_root)
                if isinstance(record, dict) and record.get("file_path")
            }
            file_forge_paths = {
                str(Path(str(record.get("file_path") or "")).resolve())
                for record in file_db.iter_file_records(path_prefix=source_root)
                if isinstance(record, dict) and record.get("file_path")
            }
            indexed_file_paths = []
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get("path"):
                    continue
                indexed_file_paths.append(_normalize_archive_entry_path(str(entry.get("path") or ""), archive_root=archive_root))
            source_tree_paths = _repo_source_file_paths(source_root)
            source_tree_unindexed = [file_path for file_path in source_tree_paths if file_path not in indexed_file_paths]
            missing_files = [file_path for file_path in indexed_file_paths if file_path not in tracked_file_paths]
            uncaptured_files = [file_path for file_path in indexed_file_paths if file_path not in captured_file_paths]
            reversible_paths = tracked_file_paths | file_forge_paths
            reversible_files = [file_path for file_path in source_tree_paths if file_path in reversible_paths]
            reversible_missing_files = [file_path for file_path in source_tree_paths if file_path not in reversible_paths]
            source_tree_unindexed_reversible = [file_path for file_path in source_tree_unindexed if file_path in file_forge_paths]
            file_record_count = len(tracked_file_paths)
            file_forge_record_count = len(file_forge_paths)
            retirement_manifest = retirement_manifests.get(repo_key) or {}
            retired = bool(retirement_manifest)
            row = {
                "repo_key": repo_key,
                "mode": mode,
                "source_root": str(source_root),
                "source_exists": source_root.exists(),
                "file_count": file_count,
                "bytes": byte_count,
                "batch_count": len(batches),
                "completed_batches": completed,
                "pending_batches": pending,
                "failed_batches": failed,
                "provenance_links": prov_links,
                "provenance_registries": prov_registry,
                "file_record_count": file_record_count,
                "file_forge_record_count": file_forge_record_count,
                "reversible_file_count": len(reversible_files),
                "reversible_missing_count": len(reversible_missing_files),
                "reversible_missing_samples": reversible_missing_files[:12],
                "source_tree_file_count": len(source_tree_paths),
                "source_tree_unindexed_count": len(source_tree_unindexed),
                "source_tree_unindexed_reversible_count": len(source_tree_unindexed_reversible),
                "source_tree_unindexed_samples": source_tree_unindexed[:12],
                "captured_file_count": len(captured_file_paths),
                "uncaptured_file_count": len(uncaptured_files),
                "uncaptured_file_samples": uncaptured_files[:12],
                "missing_file_count": len(missing_files),
                "missing_file_samples": missing_files[:12],
                "retired": retired,
                "retirement_manifest": retirement_manifest.get("manifest_path") or retirement_manifest.get("retirement_manifest_path"),
            }
            row["retirement_ready"] = _retirement_ready_for_row(row)
            row["retirement_blockers"] = _retirement_blockers_for_row(row)
            rows.append(row)
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()
        file_close = getattr(file_db, "close", None)
        if callable(file_close):
            file_close()

    rows.sort(key=lambda row: (int(row.get("bytes") or 0), str(row.get("repo_key") or "")), reverse=True)
    report = {
        "contract": "eidos.code_forge_archive_lifecycle.v1",
        "generated_at": _now_iso(),
        "repo_root": str(repo_root),
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "policy_path": str(output_dir / "repo_retention_policy.json"),
        "repo_count": len(rows),
        "repos": rows,
        "summary": {
            "ingest_and_keep": sum(1 for row in rows if row.get("mode") == "ingest_and_keep"),
            "ingest_and_remove": sum(1 for row in rows if row.get("mode") == "ingest_and_remove"),
            "retirement_ready": sum(1 for row in rows if row.get("retirement_ready")),
            "retired": sum(1 for row in rows if row.get("retired")),
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"code_forge_archive_lifecycle_{stamp}.json"
    latest_path = report_dir / "latest.json"
    _write_json(json_path, report)
    latest_path.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["latest_json"] = str(latest_path)
    return report


def _ensure_archive_plan(
    *,
    repo_root: Path,
    archive_root: Path,
    output_dir: Path,
    refresh: bool = False,
) -> dict[str, Any]:
    repo_index_path = output_dir / "repo_index.json"
    batch_plan_path = output_dir / "archive_ingestion_batches.json"
    state_path = output_dir / "archive_ingestion_state.json"
    if refresh or not (repo_index_path.exists() and batch_plan_path.exists() and state_path.exists()):
        return build_archive_plan_report(
            repo_root=repo_root,
            archive_root=archive_root,
            output_dir=output_dir,
            refresh=True,
        )
    latest_plan = _load_json(repo_root / "reports" / "code_forge_archive_plan" / "latest.json")
    if isinstance(latest_plan, dict) and latest_plan:
        return latest_plan
    return build_archive_plan_report(
        repo_root=repo_root,
        archive_root=archive_root,
        output_dir=output_dir,
        refresh=False,
    )


def run_archive_wave(
    *,
    repo_root: Path,
    archive_root: Path,
    output_dir: Path,
    repo_keys: list[str] | None,
    batch_limit: int | None,
    progress_every: int,
    retry_failed: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    archive_root = archive_root.resolve()
    output_dir = output_dir.resolve()
    db = CodeLibraryDB(repo_root / "data" / "code_forge" / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=repo_root / "data" / "code_forge" / "ingestion_runs")
    kb_path = repo_root / "data" / "knowledge_forge" / "knowledge_db.json"
    running = {
        "contract": "eidos.code_forge_archive_lifecycle.status.v1",
        "status": "running",
        "phase": "ingest_wave",
        "started_at": _now_iso(),
        "repo_keys": list(repo_keys or []),
        "output_dir": str(output_dir),
    }
    _write_json(STATUS_PATH, running)
    _append_jsonl(HISTORY_PATH, running)
    try:
        result = run_archive_ingestion_batches(
            root_path=archive_root,
            db=db,
            runner=runner,
            output_dir=output_dir,
            kb_path=kb_path,
            include_routes=["code_forge", "document_pipeline", "knowledge_metadata"],
            include_repo_keys=repo_keys or None,
            batch_limit=batch_limit,
            progress_every=max(1, int(progress_every)),
            retry_failed=bool(retry_failed),
        )
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()

    repo_index = _load_json(output_dir / "repo_index.json") or {}
    planned_repo_keys = sorted({
        str(entry.get("repo_key") or _repo_key_for_path(str(entry.get("path") or "")))
        for entry in list(repo_index.get("entries") or [])
        if isinstance(entry, dict)
    })
    selected_repo_keys = sorted({str(key) for key in (repo_keys or planned_repo_keys) if str(key).strip()})
    file_forge_results = [
        _run_file_forge_index(repo_root=repo_root, source_root=archive_root / repo_key, repo_key=repo_key)
        for repo_key in selected_repo_keys
    ]
    file_forge_summary = {
        "contract": "eidos.file_forge.archive_wave.v1",
        "generated_at": _now_iso(),
        "repo_root": str(repo_root),
        "archive_root": str(archive_root),
        "db_path": str(_file_forge_db_path(repo_root)),
        "repos": file_forge_results,
        "indexed": sum(int(item.get("indexed") or 0) for item in file_forge_results),
        "skipped": sum(int(item.get("skipped") or 0) for item in file_forge_results),
        "removed": sum(int(item.get("removed") or 0) for item in file_forge_results),
        "repo_count": len(file_forge_results),
    }
    file_forge_summary_path = output_dir / "archive_ingestion_wave_file_forge_summary.json"
    _write_json(file_forge_summary_path, file_forge_summary)

    completed = {
        "contract": "eidos.code_forge_archive_lifecycle.status.v1",
        "status": "completed",
        "phase": "ingest_wave",
        "started_at": running["started_at"],
        "finished_at": _now_iso(),
        "repo_keys": list(repo_keys or []),
        "output_dir": str(output_dir),
        "selected_batches": result.get("selected_batches"),
        "completed_batches": result.get("completed"),
        "retry_failed": bool(retry_failed),
        "failed_batches": result.get("failed"),
        "skipped_batches": result.get("skipped"),
        "summary_path": str(output_dir / "archive_ingestion_wave_summary.json"),
        "file_forge_summary_path": str(file_forge_summary_path),
        "file_forge_repo_count": int(file_forge_summary.get("repo_count") or 0),
        "file_forge_indexed": int(file_forge_summary.get("indexed") or 0),
        "file_forge_skipped": int(file_forge_summary.get("skipped") or 0),
    }
    _write_json(STATUS_PATH, completed)
    _append_jsonl(HISTORY_PATH, completed)
    return completed


def preview_retire_repos(
    *,
    repo_root: Path,
    archive_root: Path,
    output_dir: Path,
    report_dir: Path,
    repo_keys: list[str] | None,
    assume_remove_mode: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    archive_root = archive_root.resolve()
    output_dir = output_dir.resolve()
    report_dir = report_dir.resolve()
    status_report = build_repo_status_report(repo_root=repo_root, archive_root=archive_root, output_dir=output_dir, report_dir=report_dir, repo_keys=repo_keys)
    rows = list(status_report.get("repos") or [])
    if repo_keys:
        selected = {str(key) for key in repo_keys}
        rows = [row for row in rows if str(row.get("repo_key") or "") in selected]
    db = CodeLibraryDB(repo_root / "data" / "code_forge" / "library.sqlite")
    file_db = FileLibraryDB(_file_forge_db_path(repo_root))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "retirements" / f"preview_{stamp}"
    try:
        results: list[dict[str, Any]] = []
        for row in rows:
            repo_key = str(row.get("repo_key") or "")
            source_root = Path(str(row.get("source_root") or archive_root / repo_key)).resolve()
            effective_mode = str(row.get("mode") or "ingest_and_keep")
            if assume_remove_mode and effective_mode != "ingest_and_remove":
                effective_mode = "ingest_and_remove"
            ready = _retirement_ready_for_row(row, effective_mode=effective_mode)
            rec: dict[str, Any] = {
                "repo_key": repo_key,
                "current_mode": row.get("mode"),
                "effective_mode": effective_mode,
                "assume_remove_mode": bool(assume_remove_mode),
                "source_root": str(source_root),
                "source_exists": source_root.exists(),
                "retired": bool(row.get("retired")),
                "retirement_ready": ready,
            }
            if not ready:
                rec["status"] = "skipped"
                rec["reason"] = "repo is not yet retirement-ready under the effective mode"
                rec["blockers"] = _retirement_blockers_for_row(row, effective_mode=effective_mode)
                rec["captured_file_count"] = int(row.get("captured_file_count") or 0)
                rec["uncaptured_file_count"] = int(row.get("uncaptured_file_count") or 0)
                rec["uncaptured_file_samples"] = list(row.get("uncaptured_file_samples") or [])
                rec["missing_file_count"] = int(row.get("missing_file_count") or 0)
                rec["missing_file_samples"] = list(row.get("missing_file_samples") or [])
                results.append(rec)
                continue
            reconstruction_root = run_dir / "reconstruction" / repo_key
            parity_path = run_dir / "parity" / repo_key / "parity_report.json"
            manifest = _build_unified_reconstruction(
                repo_root=repo_root,
                code_db=db,
                file_db=file_db,
                source_root=source_root,
                output_dir=reconstruction_root,
                strict=True,
            )
            parity = compare_tree_parity(
                source_root=source_root,
                reconstructed_root=reconstruction_root,
                report_path=parity_path,
            )
            rec["reconstruction_manifest_path"] = manifest.get("manifest_path")
            rec["parity_report_path"] = str(parity_path)
            rec["parity_pass"] = bool(parity.get("pass"))
            rec["status"] = "ready" if rec["parity_pass"] else "failed"
            rec["would_retire"] = bool(rec["parity_pass"])
            rec["retired_root_candidate"] = str(run_dir / "stored" / repo_key)
            results.append(rec)
        payload = {
            "contract": "eidos.code_forge_archive_preview.v1",
            "generated_at": _now_iso(),
            "repo_root": str(repo_root),
            "archive_root": str(archive_root),
            "output_dir": str(output_dir),
            "assume_remove_mode": bool(assume_remove_mode),
            "retirements": results,
        }
        _write_json(run_dir / "retirement_preview.json", payload)
        return payload
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()
        file_close = getattr(file_db, "close", None)
        if callable(file_close):
            file_close()


def retire_repos(
    *,
    repo_root: Path,
    archive_root: Path,
    output_dir: Path,
    report_dir: Path,
    repo_keys: list[str] | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    archive_root = archive_root.resolve()
    output_dir = output_dir.resolve()
    report_dir = report_dir.resolve()
    status_report = build_repo_status_report(repo_root=repo_root, archive_root=archive_root, output_dir=output_dir, report_dir=report_dir, repo_keys=repo_keys)
    rows = list(status_report.get("repos") or [])
    if repo_keys:
        selected = {str(key) for key in repo_keys}
        rows = [row for row in rows if str(row.get("repo_key") or "") in selected]
    db = CodeLibraryDB(repo_root / "data" / "code_forge" / "library.sqlite")
    file_db = FileLibraryDB(_file_forge_db_path(repo_root))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "retirements" / stamp
    retire_store = run_dir / "stored"
    try:
        results: list[dict[str, Any]] = []
        for row in rows:
            repo_key = str(row.get("repo_key") or "")
            source_root = Path(str(row.get("source_root") or archive_root / repo_key)).resolve()
            rec: dict[str, Any] = {
                "repo_key": repo_key,
                "mode": row.get("mode"),
                "source_root": str(source_root),
                "dry_run": bool(dry_run),
                "retirement_ready": bool(row.get("retirement_ready")),
            }
            if not row.get("retirement_ready"):
                rec["status"] = "skipped"
                rec["reason"] = "repo is not yet retirement-ready"
                rec["blockers"] = list(row.get("retirement_blockers") or [])
                rec["captured_file_count"] = int(row.get("captured_file_count") or 0)
                rec["uncaptured_file_count"] = int(row.get("uncaptured_file_count") or 0)
                rec["uncaptured_file_samples"] = list(row.get("uncaptured_file_samples") or [])
                rec["missing_file_count"] = int(row.get("missing_file_count") or 0)
                rec["missing_file_samples"] = list(row.get("missing_file_samples") or [])
                results.append(rec)
                continue
            reconstruction_root = run_dir / "reconstruction" / repo_key
            parity_path = run_dir / "parity" / repo_key / "parity_report.json"
            manifest = _build_unified_reconstruction(
                repo_root=repo_root,
                code_db=db,
                file_db=file_db,
                source_root=source_root,
                output_dir=reconstruction_root,
                strict=True,
            )
            parity = compare_tree_parity(
                source_root=source_root,
                reconstructed_root=reconstruction_root,
                report_path=parity_path,
            )
            rec["reconstruction_manifest_path"] = manifest.get("manifest_path")
            rec["parity_report_path"] = str(parity_path)
            rec["parity_pass"] = bool(parity.get("pass"))
            if not parity.get("pass"):
                rec["status"] = "failed"
                rec["reason"] = "reconstruction parity failed"
                results.append(rec)
                continue
            retired_root = retire_store / repo_key
            if not dry_run:
                retired_root.parent.mkdir(parents=True, exist_ok=True)
                if source_root.exists():
                    shutil.move(str(source_root), str(retired_root))
            rec["status"] = "retired" if not dry_run else "dry_run"
            rec["retired_root"] = str(retired_root)
            rec["restore_command"] = f"./eidosian_venv/bin/python scripts/code_forge_archive_lifecycle.py restore --repo-key {repo_key}"
            results.append(rec)
        payload = {
            "contract": "eidos.code_forge_archive_retire.v1",
            "generated_at": _now_iso(),
            "repo_root": str(repo_root),
            "archive_root": str(archive_root),
            "output_dir": str(output_dir),
            "dry_run": bool(dry_run),
            "retirements": results,
        }
        _write_json(run_dir / "retirement_manifest.json", payload)
        RETIREMENTS_DIR.mkdir(parents=True, exist_ok=True)
        RETIREMENTS_LATEST.write_text((run_dir / "retirement_manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
        _append_jsonl(RETIREMENTS_HISTORY, payload)
        return payload
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()
        file_close = getattr(file_db, "close", None)
        if callable(file_close):
            file_close()


def restore_repo(
    *,
    repo_root: Path,
    archive_root: Path,
    output_dir: Path,
    repo_key: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    archive_root = archive_root.resolve()
    output_dir = output_dir.resolve()
    manifests = _retirement_manifests(output_dir)
    manifest = manifests.get(repo_key)
    if not isinstance(manifest, dict):
        raise FileNotFoundError(f"no retirement manifest found for {repo_key}")
    source_root = Path(str(manifest.get("source_root") or archive_root / repo_key)).resolve()
    retired_root = Path(str(manifest.get("retired_root") or "")).resolve() if manifest.get("retired_root") else None
    restored = {
        "contract": "eidos.code_forge_archive_restore.v1",
        "generated_at": _now_iso(),
        "repo_key": repo_key,
        "source_root": str(source_root),
        "restored_from_retired_root": None,
        "reconstructed": False,
    }
    if retired_root and retired_root.exists() and not source_root.exists():
        source_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(retired_root), str(source_root))
        restored["restored_from_retired_root"] = str(retired_root)
    elif not source_root.exists():
        db = CodeLibraryDB(repo_root / "data" / "code_forge" / "library.sqlite")
        file_db = FileLibraryDB(_file_forge_db_path(repo_root))
        try:
            reconstruction_root = output_dir / "restored" / repo_key
            _build_unified_reconstruction(
                repo_root=repo_root,
                code_db=db,
                file_db=file_db,
                source_root=source_root,
                output_dir=reconstruction_root,
                strict=True,
            )
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()
            file_close = getattr(file_db, "close", None)
            if callable(file_close):
                file_close()
        restored["reconstructed"] = True
        restored["reconstruction_output_dir"] = str(reconstruction_root)
    _write_json(output_dir / "retirements" / f"restore_{repo_key}.json", restored)
    return restored


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Code Forge archive ingestion lifecycle modes and reversible retirement.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", default=str(FORGE_ROOT))
    common.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    common.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    common.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))

    status_p = subparsers.add_parser("status", parents=[common])
    status_p.add_argument("--repo-key", action="append", default=[])
    status_p.add_argument("--refresh", action="store_true")
    status_p.set_defaults(func="status")

    set_mode_p = subparsers.add_parser("set-mode", parents=[common])
    set_mode_p.add_argument("--repo-key", required=True)
    set_mode_p.add_argument("--mode", required=True, choices=["ingest_and_keep", "ingest_and_remove"])
    set_mode_p.add_argument("--reason", default="")
    set_mode_p.set_defaults(func="set_mode")

    wave_p = subparsers.add_parser("run-wave", parents=[common])
    wave_p.add_argument("--repo-key", action="append", default=[])
    wave_p.add_argument("--batch-limit", type=int, default=None)
    wave_p.add_argument("--progress-every", type=int, default=200)
    wave_p.add_argument("--refresh", action="store_true")
    wave_p.add_argument("--retry-failed", action="store_true")
    wave_p.set_defaults(func="run_wave")

    preview_p = subparsers.add_parser("preview-retire", parents=[common])
    preview_p.add_argument("--repo-key", action="append", default=[])
    preview_p.add_argument("--assume-remove-mode", action="store_true")
    preview_p.add_argument("--refresh", action="store_true")
    preview_p.set_defaults(func="preview_retire")

    retire_p = subparsers.add_parser("retire", parents=[common])
    retire_p.add_argument("--repo-key", action="append", default=[])
    retire_p.add_argument("--dry-run", action="store_true")
    retire_p.add_argument("--refresh", action="store_true")
    retire_p.set_defaults(func="retire")

    restore_p = subparsers.add_parser("restore", parents=[common])
    restore_p.add_argument("--repo-key", required=True)
    restore_p.set_defaults(func="restore")

    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    archive_root = Path(args.archive_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_dir = Path(args.report_dir).resolve()

    running = {
        "contract": "eidos.code_forge_archive_lifecycle.status.v1",
        "status": "running",
        "started_at": _now_iso(),
        "phase": args.func,
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
    }
    _write_runtime_status(running)
    try:
        if args.func == "status":
            _ensure_archive_plan(
                repo_root=repo_root,
                archive_root=archive_root,
                output_dir=output_dir,
                refresh=bool(args.refresh),
            )
            payload = build_repo_status_report(repo_root=repo_root, archive_root=archive_root, output_dir=output_dir, report_dir=report_dir, repo_keys=list(args.repo_key or []))
            selected = {str(item) for item in list(args.repo_key or []) if str(item).strip()}
            if selected:
                payload = dict(payload)
                payload["repos"] = [row for row in list(payload.get("repos") or []) if str(row.get("repo_key") or "") in selected]
                payload["repo_count"] = len(payload["repos"])
                payload["summary"] = {
                    "ingest_and_keep": sum(1 for row in payload["repos"] if row.get("mode") == "ingest_and_keep"),
                    "ingest_and_remove": sum(1 for row in payload["repos"] if row.get("mode") == "ingest_and_remove"),
                    "retirement_ready": sum(1 for row in payload["repos"] if row.get("retirement_ready")),
                    "retired": sum(1 for row in payload["repos"] if row.get("retired")),
                }
                payload["selected_repo_keys"] = sorted(selected)
        elif args.func == "set_mode":
            payload = set_repo_mode(output_dir / "repo_retention_policy.json", args.repo_key, args.mode, args.reason)
        elif args.func == "run_wave":
            _ensure_archive_plan(
                repo_root=repo_root,
                archive_root=archive_root,
                output_dir=output_dir,
                refresh=bool(args.refresh),
            )
            payload = run_archive_wave(
                repo_root=repo_root,
                archive_root=archive_root,
                output_dir=output_dir,
                repo_keys=list(args.repo_key or []),
                batch_limit=args.batch_limit,
                progress_every=args.progress_every,
                retry_failed=bool(args.retry_failed),
            )
        elif args.func == "preview_retire":
            _ensure_archive_plan(
                repo_root=repo_root,
                archive_root=archive_root,
                output_dir=output_dir,
                refresh=bool(args.refresh),
            )
            payload = preview_retire_repos(
                repo_root=repo_root,
                archive_root=archive_root,
                output_dir=output_dir,
                report_dir=report_dir,
                repo_keys=list(args.repo_key or []),
                assume_remove_mode=bool(args.assume_remove_mode),
            )
        elif args.func == "retire":
            _ensure_archive_plan(
                repo_root=repo_root,
                archive_root=archive_root,
                output_dir=output_dir,
                refresh=bool(args.refresh),
            )
            payload = retire_repos(
                repo_root=repo_root,
                archive_root=archive_root,
                output_dir=output_dir,
                report_dir=report_dir,
                repo_keys=list(args.repo_key or []),
                dry_run=bool(args.dry_run),
            )
        else:
            payload = restore_repo(
                repo_root=repo_root,
                archive_root=archive_root,
                output_dir=output_dir,
                repo_key=str(args.repo_key),
            )

        completed = {
            "contract": "eidos.code_forge_archive_lifecycle.status.v1",
            "status": "completed",
            "started_at": running["started_at"],
            "finished_at": _now_iso(),
            "phase": args.func,
            "archive_root": str(archive_root),
            "output_dir": str(output_dir),
            "report_dir": str(report_dir),
            "repo_count": int(payload.get("repo_count") or len(payload.get("repos") or [])) if isinstance(payload, dict) else None,
            "selected_repo_keys": list(args.repo_key or []) if hasattr(args, "repo_key") else [],
        }
        _write_runtime_status(completed)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failed = {
            "contract": "eidos.code_forge_archive_lifecycle.status.v1",
            "status": "error",
            "started_at": running["started_at"],
            "finished_at": _now_iso(),
            "phase": args.func,
            "archive_root": str(archive_root),
            "output_dir": str(output_dir),
            "report_dir": str(report_dir),
            "error": str(exc),
        }
        _write_runtime_status(failed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
