#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _runtime_benchmark_rows(runtime_root: Path, repo_root: Path, bundle_root: Path, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    benchmark_root = runtime_root / "external_benchmarks"
    if not benchmark_root.exists():
        return rows
    for status_path in sorted(benchmark_root.glob("**/status.json"), reverse=True)[:6]:
        payload = _load_json(status_path)
        if not payload:
            continue
        run_dir = status_path.parent
        bundle_dir = bundle_root / "runtime_benchmarks" / run_dir.relative_to(benchmark_root)
        copied: list[str] = []
        for extra_name in ("status.json", "attempts.jsonl", "model_trace.jsonl", "policy.json"):
            src = run_dir / extra_name
            if not src.exists():
                continue
            target = bundle_dir / extra_name
            _copy(src, target)
            copied.append(_rel(target, bundle_root))
            files.append(
                {
                    "label": f"runtime_benchmark:{run_dir.name}:{extra_name}",
                    "source": _rel(src, repo_root),
                    "bundle_path": _rel(target, bundle_root),
                }
            )
        rows.append(
            {
                "scenario": payload.get("scenario") or run_dir.parent.name,
                "engine": payload.get("engine"),
                "model": payload.get("model"),
                "status": payload.get("status"),
                "stop_reason": payload.get("stop_reason"),
                "completed_count": _safe_int(payload.get("completed_count")),
                "attempt_count": _safe_int(payload.get("attempt_count")),
                "updated_at": payload.get("generated_at"),
                "bundle_files": copied,
            }
        )
    return rows


def export_bundle(repo_root: Path, output_root: Path) -> dict[str, Any]:
    stamp = _now_stamp()
    bundle_root = output_root / stamp
    bundle_root.mkdir(parents=True, exist_ok=True)

    proof_root = repo_root / "reports" / "proof"
    docs_root = repo_root / "docs"
    benchmarks_root = repo_root / "reports" / "external_benchmarks"
    runtime_root = repo_root / "data" / "runtime"

    files: list[dict[str, Any]] = []
    missing: list[str] = []

    def include(src: Path, relative_target: str, label: str) -> None:
        if not src.exists():
            missing.append(label)
            return
        target = bundle_root / relative_target
        _copy(src, target)
        files.append(
            {
                "label": label,
                "source": _rel(src, repo_root),
                "bundle_path": relative_target,
            }
        )

    include(proof_root / "entity_proof_scorecard_latest.json", "proof/entity_proof_scorecard_latest.json", "proof_json")
    include(proof_root / "entity_proof_scorecard_latest.md", "proof/entity_proof_scorecard_latest.md", "proof_markdown")
    include(
        proof_root / "migration_replay_scorecard_latest.json",
        "proof/migration_replay_scorecard_latest.json",
        "migration_json",
    )
    include(
        proof_root / "migration_replay_scorecard_latest.md",
        "proof/migration_replay_scorecard_latest.md",
        "migration_markdown",
    )
    include(
        proof_root / "identity_continuity_scorecard_latest.json",
        "proof/identity_continuity_scorecard_latest.json",
        "identity_continuity_json",
    )
    include(
        proof_root / "identity_continuity_scorecard_latest.md",
        "proof/identity_continuity_scorecard_latest.md",
        "identity_continuity_markdown",
    )
    recent_identity_history: list[dict[str, Any]] = []
    history_paths = [
        path
        for path in sorted(proof_root.glob("identity_continuity_scorecard_*.json"), reverse=True)
        if path.name != "identity_continuity_scorecard_latest.json"
    ]
    for path in history_paths[:5]:
        target = f"proof/identity_history/{path.name}"
        include(path, target, f"identity_history:{path.stem}")
        payload = _load_json(path)
        recent_identity_history.append(
            {
                "path": target,
                "generated_at": payload.get("generated_at"),
                "overall_score": payload.get("overall_score"),
                "status": payload.get("status"),
            }
        )
    include(docs_root / "THEORY_OF_OPERATION.md", "docs/THEORY_OF_OPERATION.md", "theory_of_operation")
    include(runtime_root / "session_bridge" / "latest_context.json", "runtime/session_bridge/latest_context.json", "session_bridge_context")
    include(runtime_root / "session_bridge" / "import_status.json", "runtime/session_bridge/import_status.json", "session_bridge_import_status")
    include(repo_root / "doc_forge" / "runtime" / "processor_status.json", "runtime/doc_processor/status.json", "doc_processor_status")
    include(repo_root / "doc_forge" / "runtime" / "processor_history.jsonl", "runtime/doc_processor/history.jsonl", "doc_processor_history")
    include(runtime_root / "qwenchat" / "status.json", "runtime/qwenchat/status.json", "qwenchat_status")
    include(runtime_root / "qwenchat" / "history.jsonl", "runtime/qwenchat/history.jsonl", "qwenchat_history")
    include(runtime_root / "living_pipeline_status.json", "runtime/living_pipeline_status.json", "living_pipeline_status")
    include(runtime_root / "living_pipeline_history.jsonl", "runtime/living_pipeline_history.jsonl", "living_pipeline_history")

    benchmark_rows: list[dict[str, Any]] = []
    if benchmarks_root.exists():
        for latest in sorted(benchmarks_root.glob("*/latest.json")):
            suite = latest.parent.name
            target = bundle_root / "external_benchmarks" / suite / "latest.json"
            _copy(latest, target)
            payload = _load_json(latest)
            row = {
                "suite": suite,
                "bundle_path": _rel(target, bundle_root),
                "source": _rel(latest, repo_root),
                "score": payload.get("score"),
                "status": payload.get("status"),
                "participant": payload.get("participant"),
                "execution_mode": payload.get("execution_mode"),
            }
            benchmark_rows.append(row)
            files.append(
                {
                    "label": f"external_benchmark:{suite}",
                    "source": _rel(latest, repo_root),
                    "bundle_path": _rel(target, bundle_root),
                }
            )
    else:
        missing.append("external_benchmarks_root")
    runtime_benchmark_rows = _runtime_benchmark_rows(runtime_root, repo_root, bundle_root, files)

    proof_payload = _load_json(proof_root / "entity_proof_scorecard_latest.json")
    migration_payload = _load_json(proof_root / "migration_replay_scorecard_latest.json")
    identity_payload = _load_json(proof_root / "identity_continuity_scorecard_latest.json")
    session_context_payload = _load_json(runtime_root / "session_bridge" / "latest_context.json")
    session_import_payload = _load_json(runtime_root / "session_bridge" / "import_status.json")
    lib_root = repo_root / "lib"
    if str(lib_root) not in sys.path:
        sys.path.insert(0, str(lib_root))
    try:
        from eidosian_runtime.session_bridge import summarize_import_status  # type: ignore

        session_bridge_summary = summarize_import_status(session_import_payload)
    except Exception:
        codex_threads = (
            (session_import_payload.get("codex") or {}).get("threads")
            if isinstance((session_import_payload.get("codex") or {}), dict)
            else {}
        )
        gemini_imported = (
            (session_import_payload.get("gemini") or {}).get("imported_ids")
            if isinstance((session_import_payload.get("gemini") or {}), dict)
            else []
        )
        session_bridge_summary = {
            "last_sync_at": session_import_payload.get("last_sync_at"),
            "codex_records": len(codex_threads) if isinstance(codex_threads, dict) else 0,
            "gemini_records": len(gemini_imported) if isinstance(gemini_imported, list) else 0,
            "codex_thread_count": len(codex_threads) if isinstance(codex_threads, dict) else 0,
            "codex_last_imported_count": _safe_int((session_import_payload.get("codex") or {}).get("last_imported_count")),
        }
        session_bridge_summary["imported_records"] = (
            session_bridge_summary["codex_records"] + session_bridge_summary["gemini_records"]
        )
    session_bridge_summary["recent_sessions"] = (
        len(session_context_payload.get("recent_sessions") or [])
        if isinstance(session_context_payload.get("recent_sessions"), list)
        else 0
    )
    doc_processor_status = _load_json(repo_root / "doc_forge" / "runtime" / "processor_status.json")
    qwenchat_status = _load_json(runtime_root / "qwenchat" / "status.json")
    living_pipeline_status = _load_json(runtime_root / "living_pipeline_status.json")

    manifest = {
        "contract": "eidos.entity_proof_bundle.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": ".",
        "bundle_root": _rel(bundle_root, output_root),
        "proof_summary": proof_payload.get("overall", {}),
        "migration_summary": {
            "overall_score": migration_payload.get("overall_score"),
            "status": migration_payload.get("status"),
        },
        "identity_summary": {
            "overall_score": identity_payload.get("overall_score"),
            "status": identity_payload.get("status"),
            "history": identity_payload.get("history") if isinstance(identity_payload.get("history"), dict) else {},
            "recent_history": recent_identity_history,
        },
        "session_bridge_summary": session_bridge_summary,
        "runtime_service_summary": {
            "doc_processor_status": doc_processor_status.get("status"),
            "doc_processor_phase": doc_processor_status.get("phase"),
            "qwenchat_status": qwenchat_status.get("status"),
            "qwenchat_phase": qwenchat_status.get("phase"),
            "living_pipeline_status": living_pipeline_status.get("status"),
            "living_pipeline_phase": living_pipeline_status.get("phase"),
        },
        "benchmarks": benchmark_rows,
        "runtime_benchmarks": runtime_benchmark_rows,
        "files": files,
        "missing": sorted(set(missing)),
    }

    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    tar_path = output_root / f"entity_proof_bundle_{stamp}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(bundle_root, arcname=stamp)

    latest_manifest = output_root / "latest_manifest.json"
    latest_bundle = output_root / "latest_bundle.txt"
    latest_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    latest_bundle.write_text(_rel(tar_path, repo_root) + "\n", encoding="utf-8")

    return {
        "manifest": _rel(manifest_path, repo_root),
        "bundle": _rel(tar_path, repo_root),
        "latest_manifest": _rel(latest_manifest, repo_root),
        "latest_bundle": _rel(latest_bundle, repo_root),
        "proof_status": manifest["proof_summary"].get("status"),
        "proof_score": manifest["proof_summary"].get("score"),
        "benchmark_count": len(benchmark_rows),
        "missing": manifest["missing"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a publishable Eidos proof bundle.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-root", default=None, help="Override output directory (default: reports/proof_bundle)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else repo_root / "reports" / "proof_bundle"
    output_root.mkdir(parents=True, exist_ok=True)
    payload = export_bundle(repo_root, output_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
