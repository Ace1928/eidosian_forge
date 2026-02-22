from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

ROUNDTRIP_REQUIRED_FILES = {
    "reconstruction_manifest": "reconstructed/reconstruction_manifest.json",
    "parity_report": "parity_report.json",
    "roundtrip_summary": "roundtrip_summary.json",
}

ROUNDTRIP_OPTIONAL_FILES = {
    "provenance_links": "provenance_links.json",
    "provenance_registry": "provenance_registry.json",
}


def _require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _validate_signature(payload: dict[str, Any], errors: list[str], prefix: str) -> None:
    signature = payload.get("signature")
    _require(isinstance(signature, dict), errors, f"{prefix}.signature must be object")
    if not isinstance(signature, dict):
        return

    algorithm = signature.get("algorithm")
    digest = signature.get("digest")
    payload_sha256 = signature.get("payload_sha256")
    _require(isinstance(algorithm, str), errors, f"{prefix}.signature.algorithm must be string")
    _require(isinstance(digest, str), errors, f"{prefix}.signature.digest must be string")
    _require(isinstance(payload_sha256, str), errors, f"{prefix}.signature.payload_sha256 must be string")
    if not (isinstance(algorithm, str) and isinstance(digest, str) and isinstance(payload_sha256, str)):
        return
    _require(bool(_HEX64.match(payload_sha256)), errors, f"{prefix}.signature.payload_sha256 must be 64-char hex")
    _require(bool(_HEX64.match(digest)), errors, f"{prefix}.signature.digest must be 64-char hex")

    canonical = _canonical_payload_bytes(payload)
    expected_sha = hashlib.sha256(canonical).hexdigest()
    if payload_sha256 != expected_sha:
        errors.append(f"{prefix}.signature.payload_sha256 mismatch")

    if algorithm == "sha256":
        if digest != expected_sha:
            errors.append(f"{prefix}.signature.digest mismatch for sha256")
    elif algorithm == "hmac-sha256":
        key = os.environ.get("EIDOS_CODE_FORGE_SIGNING_KEY")
        if key:
            expected_hmac = hmac.new(key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
            if digest != expected_hmac:
                errors.append(f"{prefix}.signature.digest mismatch for hmac-sha256")
    else:
        errors.append(f"{prefix}.signature.algorithm unsupported: {algorithm}")


def validate_reconstruction_manifest(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "manifest.generated_at must be string")
    _require(isinstance(payload.get("source_root"), str), errors, "manifest.source_root must be string")
    _require(isinstance(payload.get("output_dir"), str), errors, "manifest.output_dir must be string")
    _require(isinstance(payload.get("records_scanned"), int), errors, "manifest.records_scanned must be int")
    _require(isinstance(payload.get("files_written"), int), errors, "manifest.files_written must be int")
    _require(isinstance(payload.get("entries"), list), errors, "manifest.entries must be list")
    _require(isinstance(payload.get("missing_blobs"), list), errors, "manifest.missing_blobs must be list")
    _require(isinstance(payload.get("skipped"), list), errors, "manifest.skipped must be list")

    for idx, entry in enumerate((payload.get("entries") or [])[:100]):
        prefix = f"manifest.entries[{idx}]"
        _require(isinstance(entry, dict), errors, f"{prefix} must be object")
        if not isinstance(entry, dict):
            continue
        _require(isinstance(entry.get("relative_path"), str), errors, f"{prefix}.relative_path must be string")
        _require(isinstance(entry.get("content_hash"), str), errors, f"{prefix}.content_hash must be string")
        _require(isinstance(entry.get("written_hash"), str), errors, f"{prefix}.written_hash must be string")
        _require(isinstance(entry.get("bytes"), int), errors, f"{prefix}.bytes must be int")
    _validate_signature(payload, errors, "manifest")
    return errors


def validate_parity_report(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "parity.generated_at must be string")
    _require(isinstance(payload.get("source_root"), str), errors, "parity.source_root must be string")
    _require(
        isinstance(payload.get("reconstructed_root"), str),
        errors,
        "parity.reconstructed_root must be string",
    )
    _require(isinstance(payload.get("source_file_count"), int), errors, "parity.source_file_count must be int")
    _require(
        isinstance(payload.get("reconstructed_file_count"), int),
        errors,
        "parity.reconstructed_file_count must be int",
    )
    _require(
        isinstance(payload.get("missing_in_reconstruction"), list),
        errors,
        "parity.missing_in_reconstruction must be list",
    )
    _require(
        isinstance(payload.get("extra_in_reconstruction"), list),
        errors,
        "parity.extra_in_reconstruction must be list",
    )
    _require(isinstance(payload.get("hash_mismatches"), list), errors, "parity.hash_mismatches must be list")
    _require(isinstance(payload.get("pass"), bool), errors, "parity.pass must be bool")
    _validate_signature(payload, errors, "parity")
    return errors


def validate_apply_report(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "apply.generated_at must be string")
    _require(isinstance(payload.get("transaction_id"), str), errors, "apply.transaction_id must be string")
    _require(isinstance(payload.get("target_root"), str), errors, "apply.target_root must be string")
    _require(
        isinstance(payload.get("reconstructed_root"), str),
        errors,
        "apply.reconstructed_root must be string",
    )
    _require(isinstance(payload.get("parity_pass"), bool), errors, "apply.parity_pass must be bool")
    _require(
        isinstance(payload.get("require_parity_pass"), bool),
        errors,
        "apply.require_parity_pass must be bool",
    )
    _require(isinstance(payload.get("prune"), bool), errors, "apply.prune must be bool")
    _require(
        isinstance(payload.get("changed_or_new_count"), int),
        errors,
        "apply.changed_or_new_count must be int",
    )
    _require(isinstance(payload.get("removed_count"), int), errors, "apply.removed_count must be int")
    _require(isinstance(payload.get("backup_count"), int), errors, "apply.backup_count must be int")
    _require(isinstance(payload.get("changed_or_new"), list), errors, "apply.changed_or_new must be list")
    _require(isinstance(payload.get("removed"), list), errors, "apply.removed must be list")
    _require(isinstance(payload.get("noop"), bool), errors, "apply.noop must be bool")
    _validate_signature(payload, errors, "apply")
    return errors


def validate_roundtrip_summary(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "summary.generated_at must be string")
    _require(isinstance(payload.get("root_path"), str), errors, "summary.root_path must be string")
    _require(isinstance(payload.get("workspace_dir"), str), errors, "summary.workspace_dir must be string")
    _require(isinstance(payload.get("extensions"), list), errors, "summary.extensions must be list")
    _require(isinstance(payload.get("parity_pass"), bool), errors, "summary.parity_pass must be bool")
    _require(isinstance(payload.get("digest"), dict), errors, "summary.digest must be object")
    _require(isinstance(payload.get("reconstruction"), dict), errors, "summary.reconstruction must be object")
    _require(isinstance(payload.get("parity"), dict), errors, "summary.parity must be object")
    if payload.get("provenance_path") is not None:
        _require(isinstance(payload.get("provenance_path"), str), errors, "summary.provenance_path must be string")
    if payload.get("provenance_registry_path") is not None:
        _require(
            isinstance(payload.get("provenance_registry_path"), str),
            errors,
            "summary.provenance_registry_path must be string",
        )
    if payload.get("apply") is not None:
        _require(isinstance(payload.get("apply"), dict), errors, "summary.apply must be object when present")
    _validate_signature(payload, errors, "summary")
    return errors


def validate_provenance_links(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(payload.get("generated_at"), str), errors, "provenance.generated_at must be string")
    _require(isinstance(payload.get("stage"), str), errors, "provenance.stage must be string")
    _require(isinstance(payload.get("root_path"), str), errors, "provenance.root_path must be string")
    _require(isinstance(payload.get("provenance_id"), str), errors, "provenance.provenance_id must be string")
    _require(isinstance(payload.get("artifacts"), list), errors, "provenance.artifacts must be list")
    return errors


def validate_provenance_registry(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(
        isinstance(payload.get("schema_version"), str), errors, "provenance_registry.schema_version must be string"
    )
    _require(isinstance(payload.get("generated_at"), str), errors, "provenance_registry.generated_at must be string")
    _require(isinstance(payload.get("registry_id"), str), errors, "provenance_registry.registry_id must be string")
    _require(isinstance(payload.get("root_path"), str), errors, "provenance_registry.root_path must be string")
    _require(isinstance(payload.get("links"), dict), errors, "provenance_registry.links must be object")
    return errors


def validate_roundtrip_workspace(
    workspace_dir: Path,
    *,
    require_apply_report: bool = False,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    workspace_dir = Path(workspace_dir).resolve()
    report: dict[str, Any] = {
        "workspace_dir": str(workspace_dir),
        "files": {},
        "pass": True,
        "errors": [],
    }

    loaded: dict[str, dict[str, Any]] = {}
    validators = {
        "reconstruction_manifest": validate_reconstruction_manifest,
        "parity_report": validate_parity_report,
        "roundtrip_summary": validate_roundtrip_summary,
    }

    for key, relative_path in ROUNDTRIP_REQUIRED_FILES.items():
        validator = validators[key]
        path = workspace_dir / relative_path
        report_key = Path(relative_path).name
        file_report: dict[str, Any] = {"path": str(path), "exists": path.exists(), "errors": []}
        report["files"][report_key] = file_report
        if not path.exists():
            msg = "missing required artifact"
            file_report["errors"].append(msg)
            report["errors"].append(f"{report_key}: {msg}")
            report["pass"] = False
            continue
        try:
            payload = _load_json(path)
            loaded[key] = payload
            errs = validator(payload)
            if errs:
                file_report["errors"].extend(errs)
                report["errors"].extend([f"{report_key}: {e}" for e in errs])
                report["pass"] = False
        except Exception as exc:
            msg = f"failed to load/validate JSON: {exc}"
            file_report["errors"].append(msg)
            report["errors"].append(f"{report_key}: {msg}")
            report["pass"] = False

    optional_validators = {
        "provenance_links": validate_provenance_links,
        "provenance_registry": validate_provenance_registry,
    }
    for key, relative_path in ROUNDTRIP_OPTIONAL_FILES.items():
        validator = optional_validators[key]
        path = workspace_dir / relative_path
        report_key = Path(relative_path).name
        if not path.exists():
            continue
        file_report = {"path": str(path), "exists": True, "errors": []}
        report["files"][report_key] = file_report
        try:
            payload = _load_json(path)
            errs = validator(payload)
            if errs:
                file_report["errors"].extend(errs)
                report["errors"].extend([f"{report_key}: {e}" for e in errs])
                report["pass"] = False
        except Exception as exc:
            msg = f"failed to load/validate JSON: {exc}"
            file_report["errors"].append(msg)
            report["errors"].append(f"{report_key}: {msg}")
            report["pass"] = False

    summary = loaded.get("roundtrip_summary") or {}
    apply_summary = summary.get("apply") if isinstance(summary, dict) else None
    apply_report_path: Optional[Path] = None
    if isinstance(apply_summary, dict):
        tx_dir = apply_summary.get("backup_transaction_dir")
        if isinstance(tx_dir, str) and tx_dir:
            apply_report_path = Path(tx_dir) / "apply_report.json"

    if require_apply_report and apply_report_path is None:
        report["pass"] = False
        report["errors"].append("apply_report.json required but no backup_transaction_dir recorded")

    if apply_report_path is not None:
        file_report: dict[str, Any] = {
            "path": str(apply_report_path),
            "exists": apply_report_path.exists(),
            "errors": [],
        }
        report["files"]["apply_report.json"] = file_report
        if not apply_report_path.exists():
            msg = "missing apply report at backup transaction directory"
            file_report["errors"].append(msg)
            report["errors"].append(f"apply_report.json: {msg}")
            report["pass"] = False
        else:
            try:
                apply_payload = _load_json(apply_report_path)
                errs = validate_apply_report(apply_payload)
                if errs:
                    file_report["errors"].extend(errs)
                    report["errors"].extend([f"apply_report.json: {e}" for e in errs])
                    report["pass"] = False
            except Exception as exc:
                msg = f"failed to load/validate JSON: {exc}"
                file_report["errors"].append(msg)
                report["errors"].append(f"apply_report.json: {msg}")
                report["pass"] = False

    if verify_hashes:
        manifest = loaded.get("reconstruction_manifest") or {}
        reconstructed_root = workspace_dir / "reconstructed"
        for idx, entry in enumerate((manifest.get("entries") or [])[:2000]):
            if not isinstance(entry, dict):
                report["errors"].append(f"manifest.entries[{idx}]: entry must be object")
                report["pass"] = False
                continue
            rel = str(entry.get("relative_path") or "")
            expected_hash = str(entry.get("written_hash") or "")
            if not rel or not expected_hash:
                report["errors"].append(f"manifest.entries[{idx}]: missing relative_path/written_hash")
                report["pass"] = False
                continue
            target = reconstructed_root / rel
            if not target.exists() or not target.is_file():
                report["errors"].append(f"manifest.entries[{idx}]: missing reconstructed file {rel}")
                report["pass"] = False
                continue
            actual = _sha256_file(target)
            if actual != expected_hash:
                report["errors"].append(
                    f"manifest.entries[{idx}]: hash mismatch for {rel} ({actual} != {expected_hash})"
                )
                report["pass"] = False

    return report
