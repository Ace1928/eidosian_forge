from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from code_forge.library.db import CodeLibraryDB


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unit"


def sync_units_to_knowledge_forge(
    db: CodeLibraryDB,
    kb_path: Path,
    limit: int = 5000,
    min_token_count: int = 5,
    run_id: str | None = None,
    include_node_links: bool = False,
    node_links_limit: int = 200,
) -> dict[str, Any]:
    from knowledge_forge.core.graph import KnowledgeForge

    kb = KnowledgeForge(persistence_path=kb_path)
    existing: dict[str, str] = {}
    for node_id, node in kb.nodes.items():
        meta = node.metadata or {}
        unit_id = meta.get("code_unit_id")
        if unit_id:
            existing[str(unit_id)] = node_id

    units = list(db.iter_units(limit=max(1, int(limit)), run_id=run_id))
    created = 0
    updated = 0
    links = 0
    id_to_node = dict(existing)
    node_links: list[dict[str, str]] = []

    for unit in units:
        unit_id = str(unit.get("id"))
        if not unit_id:
            continue
        if int(unit.get("token_count") or 0) < max(0, int(min_token_count)):
            continue

        qn = str(unit.get("qualified_name") or unit.get("name") or unit_id)
        content = f"[CODE {unit.get('language', 'text')}:{unit.get('unit_type', 'node')}] {qn}\n"
        content += f"Path: {unit.get('file_path')}:{unit.get('line_start')}\n"
        content += f"Token count: {unit.get('token_count', 0)}\n"
        if unit.get("content_hash"):
            blob = db.get_text(str(unit["content_hash"]))
            if blob:
                content += "Snippet:\n" + blob[:1800]

        concepts = [
            "code",
            str(unit.get("language") or "text"),
            str(unit.get("unit_type") or "node"),
            str(unit.get("name") or "unit"),
        ]
        tags = [
            "code",
            "code_forge",
            str(unit.get("language") or "text"),
            str(unit.get("unit_type") or "node"),
        ]
        metadata = {
            "source": "code_forge",
            "code_unit_id": unit_id,
            "qualified_name": qn,
            "file_path": unit.get("file_path"),
            "line_start": unit.get("line_start"),
            "line_end": unit.get("line_end"),
            "token_count": unit.get("token_count"),
            "normalized_hash": unit.get("normalized_hash"),
            "simhash64": unit.get("simhash64"),
        }

        if unit_id in existing:
            # KnowledgeForge does not provide update API; keep prior node and skip duplicates.
            updated += 1
            node_id = existing[unit_id]
            link_status = "existing"
        else:
            node = kb.add_knowledge(content=content, concepts=concepts, tags=tags, metadata=metadata)
            node_id = node.id
            id_to_node[unit_id] = node_id
            created += 1
            link_status = "created"

        if include_node_links and len(node_links) < max(1, int(node_links_limit)):
            node_links.append(
                {
                    "unit_id": unit_id,
                    "node_id": str(node_id),
                    "status": link_status,
                }
            )

        parent = unit.get("parent_id")
        if parent and str(parent) in id_to_node and str(parent) != unit_id:
            kb.link_nodes(id_to_node[str(parent)], node_id)
            links += 1

    kb.save()
    return {
        "kb_path": str(kb_path),
        "run_id": run_id,
        "scanned_units": len(units),
        "created_nodes": created,
        "existing_nodes": updated,
        "links_created": links,
        "node_links": node_links if include_node_links else [],
        "node_links_truncated": bool(include_node_links and len(units) > len(node_links)),
    }


def export_units_for_graphrag(
    db: CodeLibraryDB,
    output_dir: Path,
    limit: int = 20000,
    min_token_count: int = 5,
    run_id: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    by_language: dict[str, int] = {}
    documents: list[dict[str, Any]] = []

    for unit in db.iter_units(limit=max(1, int(limit)), run_id=run_id):
        if int(unit.get("token_count") or 0) < max(0, int(min_token_count)):
            skipped += 1
            continue

        unit_id = str(unit.get("id"))
        qn = str(unit.get("qualified_name") or unit.get("name") or unit_id)
        language = str(unit.get("language") or "text")
        unit_type = str(unit.get("unit_type") or "node")

        text = [
            f"Code Unit: {qn}",
            f"Language: {language}",
            f"Type: {unit_type}",
            f"Path: {unit.get('file_path')}:{unit.get('line_start')}",
            f"Token count: {unit.get('token_count', 0)}",
            "",
        ]

        if unit.get("content_hash"):
            blob = db.get_text(str(unit["content_hash"]))
            if blob:
                text.append("Snippet:")
                text.append(blob[:4000])

        fname = f"{language}_{_safe_name(qn)}_{unit_id[:8]}.txt"
        target = output_dir / fname
        target.write_text("\n".join(text).strip() + "\n", encoding="utf-8")
        exported += 1
        by_language[language] = by_language.get(language, 0) + 1
        documents.append(
            {
                "unit_id": unit_id,
                "qualified_name": qn,
                "language": language,
                "unit_type": unit_type,
                "source_file_path": str(unit.get("file_path") or ""),
                "document_path": str(target.resolve()),
            }
        )

    manifest = {
        "output_dir": str(output_dir.resolve()),
        "run_id": run_id,
        "generated_documents": exported,
        "skipped_documents": skipped,
        "by_language": by_language,
        "documents": documents,
    }
    manifest_path = output_dir / "graphrag_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "run_id": run_id,
        "exported": exported,
        "skipped": skipped,
        "by_language": by_language,
        "manifest_path": str(manifest_path.resolve()),
    }
