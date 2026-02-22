from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from code_forge.analyzer.generic_analyzer import GenericCodeAnalyzer
from code_forge.digester.pipeline import run_archive_digester
from code_forge.ingest.runner import IngestionRunner
from code_forge.integration.provenance import write_provenance_links
from code_forge.integration.provenance_registry import (
    load_latest_benchmark_for_root,
    write_provenance_registry,
)
from code_forge.library.db import CodeLibraryDB


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_tree_hashes(
    root: Path,
    *,
    extensions: Optional[Iterable[str]] = None,
    exclude_relative_paths: Optional[set[str]] = None,
) -> dict[str, str]:
    root = Path(root).resolve()
    ext_set = {str(ext).lower() for ext in extensions} if extensions else None
    out: dict[str, str] = {}
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root).as_posix()
        if exclude_relative_paths and rel in exclude_relative_paths:
            continue
        suffix = file_path.suffix.lower()
        if ext_set is not None and suffix not in ext_set:
            continue
        out[rel] = _sha256_file(file_path)
    return out


def _collect_selected_hashes(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    root = Path(root).resolve()
    out: dict[str, str] = {}
    for rel in sorted(set(str(p) for p in relative_paths if str(p))):
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            out[rel] = _sha256_file(candidate)
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    # Signature is computed over canonical JSON without the signature envelope.
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _attach_artifact_signature(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = _canonical_payload_bytes(payload)
    payload_sha256 = _sha256_bytes(canonical)
    signing_key = os.environ.get("EIDOS_CODE_FORGE_SIGNING_KEY")
    key_id = os.environ.get("EIDOS_CODE_FORGE_SIGNING_KEY_ID")
    algorithm = "sha256"
    digest = payload_sha256
    if signing_key:
        algorithm = "hmac-sha256"
        digest = hmac.new(signing_key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
    signed = dict(payload)
    signed["signature"] = {
        "algorithm": algorithm,
        "digest": digest,
        "payload_sha256": payload_sha256,
        "key_id": key_id if key_id else None,
        "signed_at": _utc_now(),
    }
    return signed


def build_reconstruction_from_library(
    *,
    db: CodeLibraryDB,
    source_root: Path,
    output_dir: Path,
    extensions: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    strict: bool = True,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ext_set = {str(ext).lower() for ext in extensions} if extensions else None

    records = list(
        db.iter_file_records(
            path_prefix=str(source_root),
            limit=max_files if max_files is not None else 500000,
        )
    )
    entries: list[dict[str, Any]] = []
    missing_blobs: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    for record in records:
        abs_path = Path(str(record.get("file_path") or "")).resolve()
        try:
            rel_path = abs_path.relative_to(source_root).as_posix()
        except ValueError:
            skipped.append(
                {
                    "file_path": str(abs_path),
                    "reason": "outside_source_root",
                }
            )
            continue

        suffix = abs_path.suffix.lower()
        if ext_set is not None and suffix not in ext_set:
            skipped.append(
                {
                    "file_path": str(abs_path),
                    "reason": "filtered_by_extension",
                }
            )
            continue

        content_hash = str(record.get("content_hash") or "")
        content = db.get_text(content_hash) if content_hash else None
        if content is None:
            missing_blobs.append(
                {
                    "file_path": str(abs_path),
                    "content_hash": content_hash,
                }
            )
            continue

        target = output_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = content.encode("utf-8")
        computed_hash = _sha256_bytes(payload)
        target.write_bytes(payload)

        entries.append(
            {
                "relative_path": rel_path,
                "content_hash": content_hash,
                "written_hash": computed_hash,
                "bytes": len(payload),
                "analysis_version": int(record.get("analysis_version") or 0),
                "updated_at": str(record.get("updated_at") or ""),
            }
        )

    manifest = {
        "generated_at": _utc_now(),
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "strict_mode": bool(strict),
        "selected_extensions": sorted(ext_set) if ext_set else None,
        "records_scanned": len(records),
        "files_written": len(entries),
        "missing_blob_count": len(missing_blobs),
        "skipped_count": len(skipped),
        "entries": entries,
        "missing_blobs": missing_blobs,
        "skipped": skipped,
    }

    manifest = _attach_artifact_signature(manifest)
    manifest_path = output_dir / "reconstruction_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    if strict and missing_blobs:
        raise RuntimeError(f"missing content blobs for {len(missing_blobs)} files")
    return manifest


def compare_tree_parity(
    *,
    source_root: Path,
    reconstructed_root: Path,
    report_path: Optional[Path] = None,
    extensions: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    reconstructed_root = Path(reconstructed_root).resolve()
    roundtrip_meta = {
        "reconstruction_manifest.json",
        "parity_report.json",
        "roundtrip_summary.json",
    }
    src = _collect_tree_hashes(source_root, extensions=extensions)
    dst = _collect_tree_hashes(
        reconstructed_root,
        extensions=extensions,
        exclude_relative_paths=roundtrip_meta,
    )

    src_paths = set(src.keys())
    dst_paths = set(dst.keys())
    missing_in_reconstruction = sorted(src_paths - dst_paths)
    extra_in_reconstruction = sorted(dst_paths - src_paths)
    mismatched: list[dict[str, str]] = []
    for rel in sorted(src_paths & dst_paths):
        if src[rel] != dst[rel]:
            mismatched.append(
                {
                    "relative_path": rel,
                    "source_hash": src[rel],
                    "reconstructed_hash": dst[rel],
                }
            )

    parity = {
        "generated_at": _utc_now(),
        "source_root": str(source_root),
        "reconstructed_root": str(reconstructed_root),
        "source_file_count": len(src),
        "reconstructed_file_count": len(dst),
        "missing_in_reconstruction": missing_in_reconstruction,
        "extra_in_reconstruction": extra_in_reconstruction,
        "hash_mismatches": mismatched,
        "pass": not missing_in_reconstruction and not extra_in_reconstruction and not mismatched,
    }
    parity = _attach_artifact_signature(parity)
    if report_path is not None:
        report_path = Path(report_path).resolve()
        _write_json(report_path, parity)
        parity["report_path"] = str(report_path)
    return parity


def apply_reconstruction(
    *,
    reconstructed_root: Path,
    target_root: Path,
    backup_root: Path,
    parity_report: Optional[dict[str, Any]] = None,
    require_parity_pass: bool = True,
    prune: bool = True,
    extensions: Optional[Iterable[str]] = None,
    managed_relative_paths: Optional[Iterable[str]] = None,
    require_manifest: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    reconstructed_root = Path(reconstructed_root).resolve()
    target_root = Path(target_root).resolve()
    backup_root = Path(backup_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    backup_root.mkdir(parents=True, exist_ok=True)

    roundtrip_meta = {
        "reconstruction_manifest.json",
        "parity_report.json",
        "roundtrip_summary.json",
    }
    if parity_report is None:
        parity_report = compare_tree_parity(
            source_root=target_root,
            reconstructed_root=reconstructed_root,
        )
    if require_parity_pass and not bool(parity_report.get("pass")):
        raise RuntimeError("refusing apply: parity report is failing")

    scoped_paths = [str(p) for p in managed_relative_paths or [] if str(p)]
    if not scoped_paths:
        manifest_path = reconstructed_root / "reconstruction_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                scoped_paths = [
                    str(entry.get("relative_path") or "")
                    for entry in manifest.get("entries", [])
                    if entry.get("relative_path")
                ]
            except Exception:
                scoped_paths = []

    if require_manifest and not scoped_paths:
        raise RuntimeError("refusing apply: reconstruction manifest paths required but unavailable")

    if scoped_paths:
        target_hashes = _collect_selected_hashes(target_root, scoped_paths)
        reconstructed_hashes = _collect_selected_hashes(reconstructed_root, scoped_paths)
    else:
        target_hashes = _collect_tree_hashes(target_root, extensions=extensions)
        reconstructed_hashes = _collect_tree_hashes(
            reconstructed_root,
            extensions=extensions,
            exclude_relative_paths=roundtrip_meta,
        )

    changed_or_new = sorted(rel for rel, digest in reconstructed_hashes.items() if target_hashes.get(rel) != digest)
    removed = sorted(set(target_hashes.keys()) - set(reconstructed_hashes.keys())) if prune else []

    transaction_id = f"roundtrip_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    transaction_dir = backup_root / transaction_id
    backup_count = 0

    if (changed_or_new or removed) and not dry_run:
        transaction_dir.mkdir(parents=True, exist_ok=True)
        for rel in changed_or_new:
            src = target_root / rel
            if src.exists():
                backup_path = transaction_dir / "before" / rel
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, backup_path)
                backup_count += 1
        for rel in removed:
            src = target_root / rel
            if src.exists():
                backup_path = transaction_dir / "before" / rel
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, backup_path)
                backup_count += 1

    if not dry_run:
        for rel in changed_or_new:
            src = reconstructed_root / rel
            dst = target_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        for rel in removed:
            target = target_root / rel
            if target.exists():
                target.unlink()
                parent = target.parent
                while parent != target_root and parent.exists():
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent

    report = {
        "generated_at": _utc_now(),
        "transaction_id": transaction_id,
        "target_root": str(target_root),
        "reconstructed_root": str(reconstructed_root),
        "backup_root": str(backup_root),
        "backup_transaction_dir": str(transaction_dir) if transaction_dir.exists() else None,
        "parity_pass": bool(parity_report.get("pass")),
        "require_parity_pass": bool(require_parity_pass),
        "prune": bool(prune),
        "dry_run": bool(dry_run),
        "changed_or_new_count": len(changed_or_new),
        "removed_count": len(removed),
        "backup_count": backup_count,
        "changed_or_new": changed_or_new,
        "removed": removed,
        "noop": not changed_or_new and not removed,
    }

    report = _attach_artifact_signature(report)
    if transaction_dir.exists():
        _write_json(transaction_dir / "apply_report.json", report)
    return report


def run_roundtrip_pipeline(
    *,
    root_path: Path,
    db: CodeLibraryDB,
    runner: IngestionRunner,
    workspace_dir: Path,
    mode: str = "analysis",
    extensions: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    progress_every: int = 200,
    sync_knowledge_path: Optional[Path] = None,
    sync_memory_path: Optional[Path] = None,
    graphrag_output_dir: Optional[Path] = None,
    graph_export_limit: int = 20000,
    integration_policy: str = "effective_run",
    strict_validation: bool = True,
    apply: bool = False,
    backup_root: Optional[Path] = None,
) -> dict[str, Any]:
    root_path = Path(root_path).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    digester_dir = workspace_dir / "digester"
    reconstructed_dir = workspace_dir / "reconstructed"
    parity_path = workspace_dir / "parity_report.json"
    summary_path = workspace_dir / "roundtrip_summary.json"

    effective_extensions = (
        list(extensions) if extensions is not None else sorted(GenericCodeAnalyzer.supported_extensions())
    )

    digest = run_archive_digester(
        root_path=root_path,
        db=db,
        runner=runner,
        output_dir=digester_dir,
        mode=mode,
        extensions=effective_extensions,
        max_files=max_files,
        progress_every=progress_every,
        sync_knowledge_path=sync_knowledge_path,
        sync_memory_path=sync_memory_path,
        graphrag_output_dir=graphrag_output_dir,
        graph_export_limit=graph_export_limit,
        integration_policy=integration_policy,
        strict_validation=strict_validation,
    )

    reconstruction = build_reconstruction_from_library(
        db=db,
        source_root=root_path,
        output_dir=reconstructed_dir,
        extensions=effective_extensions,
        max_files=max_files,
        strict=strict_validation,
    )
    parity = compare_tree_parity(
        source_root=root_path,
        reconstructed_root=reconstructed_dir,
        report_path=parity_path,
        extensions=effective_extensions,
    )

    apply_report = None
    if apply:
        apply_report = apply_reconstruction(
            reconstructed_root=reconstructed_dir,
            target_root=root_path,
            backup_root=backup_root or (workspace_dir / "backups"),
            parity_report=parity,
            require_parity_pass=True,
            prune=True,
            extensions=effective_extensions,
            managed_relative_paths=[
                str(entry.get("relative_path") or "")
                for entry in reconstruction.get("entries", [])
                if entry.get("relative_path")
            ],
            require_manifest=True,
        )

    summary = {
        "generated_at": _utc_now(),
        "root_path": str(root_path),
        "workspace_dir": str(workspace_dir),
        "extensions": effective_extensions,
        "integration_policy": str(integration_policy),
        "digester_summary_path": str(digester_dir / "archive_digester_summary.json"),
        "reconstruction_manifest_path": reconstruction.get("manifest_path"),
        "parity_report_path": str(parity_path),
        "parity_pass": bool(parity.get("pass")),
        "digest": digest,
        "reconstruction": reconstruction,
        "parity": parity,
        "apply": apply_report,
    }

    provenance = write_provenance_links(
        output_path=workspace_dir / "provenance_links.json",
        stage="roundtrip",
        root_path=root_path,
        source_run_id=str((digest.get("ingestion_stats") or {}).get("run_id") or ""),
        integration_policy=str(integration_policy),
        integration_run_id=digest.get("integration_run_id"),
        artifact_paths=[
            ("digester_summary", digester_dir / "archive_digester_summary.json"),
            ("reconstruction_manifest", reconstructed_dir / "reconstruction_manifest.json"),
            ("parity_report", parity_path),
            ("roundtrip_summary", summary_path),
        ],
        knowledge_sync=digest.get("knowledge_sync"),
        memory_sync=digest.get("memory_sync"),
        graphrag_export=digest.get("graphrag_export"),
        extra={
            "parity_pass": bool(parity.get("pass")),
            "apply_enabled": bool(apply),
            "apply_noop": bool((apply_report or {}).get("noop")) if isinstance(apply_report, dict) else None,
        },
    )
    summary["provenance_path"] = provenance.get("path")
    benchmark_payload = load_latest_benchmark_for_root(
        root_path=root_path,
        search_roots=[workspace_dir, workspace_dir.parent, root_path, root_path / "reports"],
    )
    registry = write_provenance_registry(
        output_path=workspace_dir / "provenance_registry.json",
        provenance_payload=provenance,
        stage_summary_payload=summary,
        drift_payload=digest.get("drift") if isinstance(digest.get("drift"), dict) else None,
        benchmark_payload=benchmark_payload,
    )
    summary["provenance_registry_path"] = registry.get("path")
    summary = _attach_artifact_signature(summary)
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
