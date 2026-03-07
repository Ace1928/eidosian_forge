from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("\\", "/")


def _slug(value: Any, default: str = "item", *, limit: int = 48) -> str:
    text = _normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-._")
    return (text or default)[: max(8, int(limit))]


def _digest(parts: Iterable[Any], *, size: int = 20) -> str:
    payload = "|".join(_normalize_text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[: max(8, int(size))]


def build_gis_id(*, namespace: str, kind: str, parts: Iterable[Any], label: str | None = None) -> str:
    digest = _digest(parts)
    if label:
        return f"gis:eidos:{_slug(namespace)}:{_slug(kind)}:{_slug(label)}:{digest}"
    return f"gis:eidos:{_slug(namespace)}:{_slug(kind)}:{digest}"


def build_run_gis_id(*, root_path: str, mode: str, run_id: str) -> str:
    root = str(Path(root_path).as_posix())
    return build_gis_id(
        namespace="code-forge",
        kind="run",
        parts=[root, mode, run_id],
        label=f"{Path(root).name}-{mode}",
    )


def build_code_unit_gis_id(
    *,
    language: str,
    unit_type: str,
    file_path: str,
    qualified_name: str | None,
    name: str,
    line_start: int | None,
    line_end: int | None,
    content_hash: str | None,
) -> str:
    identity = qualified_name or name or Path(file_path).name
    return build_gis_id(
        namespace="code-forge",
        kind="code-unit",
        parts=[language, unit_type, file_path, identity, line_start or "", line_end or "", content_hash or ""],
        label=identity,
    )


def build_artifact_gis_id(
    *,
    stage: str,
    root_path: str,
    artifact_kind: str,
    artifact_path: str,
    provenance_id: str | None = None,
) -> str:
    return build_gis_id(
        namespace="code-forge",
        kind="artifact",
        parts=[stage, root_path, artifact_kind, artifact_path, provenance_id or ""],
        label=f"{artifact_kind}-{Path(artifact_path).name}",
    )


def build_provenance_gis_id(*, stage: str, root_path: str, provenance_id: str) -> str:
    return build_gis_id(
        namespace="code-forge",
        kind="provenance",
        parts=[stage, root_path, provenance_id],
        label=f"{stage}-provenance",
    )


def build_registry_gis_id(*, root_path: str, registry_id: str) -> str:
    return build_gis_id(
        namespace="code-forge",
        kind="provenance-registry",
        parts=[root_path, registry_id],
        label="registry",
    )
