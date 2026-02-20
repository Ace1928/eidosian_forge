from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_forge.library.db import CodeLibraryDB


def _load_existing_code_unit_links(memory_path: Path) -> dict[str, str]:
    if not memory_path.exists():
        return {}
    try:
        payload = json.loads(memory_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    out: dict[str, str] = {}
    for memory_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") or {}
        if not isinstance(meta, dict):
            continue
        unit_id = meta.get("code_unit_id")
        if not unit_id:
            continue
        out[str(unit_id)] = str(memory_id)
    return out


def sync_units_to_memory_forge(
    db: CodeLibraryDB,
    memory_path: Path,
    *,
    limit: int = 5000,
    min_token_count: int = 8,
    run_id: str | None = None,
    include_memory_links: bool = False,
    memory_links_limit: int = 200,
) -> dict[str, Any]:
    """Persist compact code-unit memories and return linkage data.

    This intentionally uses JSON-backed MemoryForge storage so it remains portable in
    Termux/Linux without embedding dependencies. Links are keyed by `code_unit_id`.
    """

    from memory_forge.core.config import BackendConfig, MemoryConfig
    from memory_forge.core.main import MemoryForge

    memory_path = Path(memory_path).resolve()
    existing = _load_existing_code_unit_links(memory_path)

    config = MemoryConfig(
        episodic=BackendConfig(type="json", connection_string=str(memory_path), collection_name="code_forge_units")
    )
    forge = MemoryForge(config=config, embedder=None)

    units = list(db.iter_units(limit=max(1, int(limit)), run_id=run_id))
    created = 0
    existing_count = 0
    links: list[dict[str, str]] = []
    links_limit = max(1, int(memory_links_limit))

    for unit in units:
        unit_id = str(unit.get("id") or "")
        if not unit_id:
            continue
        if int(unit.get("token_count") or 0) < max(0, int(min_token_count)):
            continue

        if unit_id in existing:
            existing_count += 1
            if include_memory_links and len(links) < links_limit:
                links.append({"unit_id": unit_id, "memory_id": existing[unit_id], "status": "existing"})
            continue

        qn = str(unit.get("qualified_name") or unit.get("name") or unit_id)
        content = (
            f"[CODE_MEMORY] {qn}\n"
            f"Language: {unit.get('language') or 'text'}\n"
            f"Type: {unit.get('unit_type') or 'node'}\n"
            f"Path: {unit.get('file_path')}:{unit.get('line_start')}\n"
            f"Tokens: {unit.get('token_count') or 0}"
        )
        metadata = {
            "source": "code_forge",
            "code_unit_id": unit_id,
            "qualified_name": qn,
            "file_path": unit.get("file_path"),
            "line_start": unit.get("line_start"),
            "line_end": unit.get("line_end"),
            "language": unit.get("language"),
            "unit_type": unit.get("unit_type"),
            "normalized_hash": unit.get("normalized_hash"),
            "run_id": run_id,
        }
        memory_id = str(forge.remember(content=content, metadata=metadata))
        existing[unit_id] = memory_id
        created += 1
        if include_memory_links and len(links) < links_limit:
            links.append({"unit_id": unit_id, "memory_id": memory_id, "status": "created"})

    return {
        "memory_path": str(memory_path),
        "run_id": run_id,
        "scanned_units": len(units),
        "created_memories": created,
        "existing_memories": existing_count,
        "memory_links": links if include_memory_links else [],
        "memory_links_truncated": bool(include_memory_links and len(units) > len(links)),
    }
