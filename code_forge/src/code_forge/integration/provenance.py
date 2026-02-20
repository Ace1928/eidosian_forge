from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _artifact_records(paths: Iterable[tuple[str, Path]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for artifact_kind, artifact_path in paths:
        p = Path(artifact_path).resolve()
        exists = p.exists() and p.is_file()
        out.append(
            {
                "artifact_kind": str(artifact_kind),
                "path": str(p),
                "exists": bool(exists),
                "sha256": _sha256_file(p) if exists else None,
            }
        )
    return out


def write_provenance_links(
    *,
    output_path: Path,
    stage: str,
    root_path: Path,
    source_run_id: str | None,
    integration_policy: str,
    integration_run_id: str | None,
    artifact_paths: Iterable[tuple[str, Path]],
    knowledge_sync: dict[str, Any] | None,
    graphrag_export: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _artifact_records(artifact_paths)
    digest = hashlib.sha256()
    digest.update(str(Path(root_path).resolve()).encode("utf-8"))
    digest.update(str(source_run_id or "").encode("utf-8"))
    digest.update(str(integration_policy).encode("utf-8"))
    digest.update(str(integration_run_id or "").encode("utf-8"))
    for rec in records:
        digest.update(str(rec.get("artifact_kind") or "").encode("utf-8"))
        digest.update(str(rec.get("sha256") or "").encode("utf-8"))

    kb_links = []
    if isinstance(knowledge_sync, dict):
        kb_links = list(knowledge_sync.get("node_links") or [])

    grag_docs = []
    if isinstance(graphrag_export, dict):
        manifest_path = graphrag_export.get("manifest_path")
        if isinstance(manifest_path, str) and manifest_path:
            payload = _safe_load_json(Path(manifest_path))
            if isinstance(payload, dict):
                grag_docs = list(payload.get("documents") or [])

    provenance = {
        "generated_at": _utc_now(),
        "stage": str(stage),
        "root_path": str(Path(root_path).resolve()),
        "source_run_id": source_run_id,
        "integration_policy": str(integration_policy),
        "integration_run_id": integration_run_id,
        "provenance_id": digest.hexdigest()[:24],
        "artifacts": records,
        "knowledge_links": {
            "count": len(kb_links),
            "links": kb_links,
        },
        "graphrag_links": {
            "count": len(grag_docs),
            "documents": grag_docs,
            "manifest_path": graphrag_export.get("manifest_path") if isinstance(graphrag_export, dict) else None,
        },
        "extra": extra or {},
    }
    output_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    provenance["path"] = str(output_path)
    return provenance


def read_provenance_links(path: Path) -> dict[str, Any]:
    payload = _safe_load_json(Path(path))
    if payload is None:
        raise ValueError(f"invalid provenance payload: {path}")
    return payload
