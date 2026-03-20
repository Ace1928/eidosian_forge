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
    FORGE_ROOT / "code_forge" / "src",
    FORGE_ROOT / "lib",
):
    text = str(extra)
    if extra.exists() and text not in sys.path:
        sys.path.insert(0, text)

from code_forge.digester.pipeline import run_archive_ingestion_batches  # type: ignore
from code_forge.ingest.runner import IngestionRunner  # type: ignore
from code_forge.library.db import CodeLibraryDB  # type: ignore
from code_forge.reconstruct.pipeline import build_reconstruction_from_library, compare_tree_parity  # type: ignore
from code_forge_archive_plan import build_archive_plan_report, _repo_key_for_path  # type: ignore

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


def build_repo_status_report(
    *,
    repo_root: Path,
    archive_root: Path,
    output_dir: Path,
    report_dir: Path,
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

    db = CodeLibraryDB(repo_root / "data" / "code_forge" / "library.sqlite")
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
        for repo_key in sorted(entries_by_repo):
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

            mode = _repo_mode(policy, repo_key)
            file_record_count = db.count_file_records(path_prefix=source_root)
            retirement_manifest = retirement_manifests.get(repo_key) or {}
            retired = bool(retirement_manifest)
            retirement_ready = (
                mode == "ingest_and_remove"
                and not retired
                and bool(batches)
                and completed == len(batches)
                and failed == 0
                and pending == 0
                and prov_links >= len(batches)
                and prov_registry >= len(batches)
                and file_record_count >= file_count
            )
            rows.append(
                {
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
                    "retirement_ready": retirement_ready,
                    "retired": retired,
                    "retirement_manifest": retirement_manifest.get("manifest_path") or retirement_manifest.get("retirement_manifest_path"),
                }
            )
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()

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
        )
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()
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
        "failed_batches": result.get("failed"),
        "skipped_batches": result.get("skipped"),
        "summary_path": str(output_dir / "archive_ingestion_wave_summary.json"),
    }
    _write_json(STATUS_PATH, completed)
    _append_jsonl(HISTORY_PATH, completed)
    return completed


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
    status_report = build_repo_status_report(repo_root=repo_root, archive_root=archive_root, output_dir=output_dir, report_dir=report_dir)
    rows = list(status_report.get("repos") or [])
    if repo_keys:
        selected = {str(key) for key in repo_keys}
        rows = [row for row in rows if str(row.get("repo_key") or "") in selected]
    db = CodeLibraryDB(repo_root / "data" / "code_forge" / "library.sqlite")
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
                results.append(rec)
                continue
            reconstruction_root = run_dir / "reconstruction" / repo_key
            parity_path = run_dir / "parity" / repo_key / "parity_report.json"
            manifest = build_reconstruction_from_library(
                db=db,
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
        try:
            reconstruction_root = output_dir / "restored" / repo_key
            build_reconstruction_from_library(
                db=db,
                source_root=source_root,
                output_dir=reconstruction_root,
                strict=True,
            )
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()
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
    wave_p.set_defaults(func="run_wave")

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
            payload = build_repo_status_report(repo_root=repo_root, archive_root=archive_root, output_dir=output_dir, report_dir=report_dir)
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
