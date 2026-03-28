from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TERM_RE = re.compile(r"[a-z0-9_]+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TERM_RE.finditer(text or ""):
        token = match.group(0).lower()
        if not token:
            continue
        tokens.add(token)
        for part in re.split(r"[_-]+", token):
            if part:
                tokens.add(part)
    return tokens


def _load_word_metrics(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "word_count": 0,
            "relationship_count": 0,
            "lexeme_count": 0,
            "translation_count": 0,
            "base_aligned_count": 0,
            "base_terms": [],
        }
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        words = conn.execute("SELECT term FROM words").fetchall()
        relationships = int(conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0])
        lexemes = conn.execute(
            "SELECT lemma, lang, base_term FROM lexemes ORDER BY last_refreshed DESC LIMIT 400"
        ).fetchall()
        lexeme_count = int(conn.execute("SELECT COUNT(*) FROM lexemes").fetchone()[0])
        translation_count = int(conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0])
        base_aligned_count = int(
            conn.execute("SELECT COUNT(*) FROM lexemes WHERE COALESCE(base_term, '') != ''").fetchone()[0]
        )
    word_terms = {str(row[0]).strip().lower() for row in words if str(row[0]).strip()}
    base_terms = []
    for row in lexemes:
        base_term = str(row["base_term"] or "").strip().lower()
        if base_term:
            base_terms.append(base_term)
    return {
        "word_count": len(word_terms),
        "relationship_count": relationships,
        "lexeme_count": lexeme_count,
        "translation_count": translation_count,
        "base_aligned_count": base_aligned_count,
        "word_terms": word_terms,
        "base_terms": sorted(set(base_terms)),
        "recent_lexemes": [dict(row) for row in lexemes],
    }


def _load_knowledge_tokens(kb_path: Path) -> dict[str, Any]:
    payload = _load_json(kb_path)
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), dict) else {}
    tag_tokens: set[str] = set()
    content_tokens: set[str] = set()
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
        for tag in tags:
            if isinstance(tag, str):
                tag_tokens.update(_tokenize(tag))
        content = str(node.get("content") or "")
        content_tokens.update(token for token in _tokenize(content) if len(token) >= 3)
    return {
        "node_count": len(nodes),
        "tag_tokens": tag_tokens,
        "content_tokens": content_tokens,
    }


def _load_file_tokens(repo_root: Path) -> dict[str, Any]:
    db_path = repo_root / "data" / "file_forge" / "library.sqlite"
    if not db_path.exists():
        return {
            "file_count": 0,
            "link_count": 0,
            "relationship_count": 0,
            "path_tokens": set(),
            "semantic_tokens": set(),
        }
    try:
        import sys
        file_forge_src = repo_root / "file_forge" / "src"
        lib_root = repo_root / "lib"
        if str(file_forge_src) not in sys.path:
            sys.path.insert(0, str(file_forge_src))
        if str(lib_root) not in sys.path:
            sys.path.insert(0, str(lib_root))
        from file_forge.library import FileLibraryDB  # type: ignore

        db = FileLibraryDB(db_path)
        path_tokens: set[str] = set()
        semantic_tokens: set[str] = set()
        count = 0
        link_count = 0
        relationship_count = 0
        for row in db.iter_file_records(limit=50000):
            if not isinstance(row, dict):
                continue
            path_text = str(row.get("file_path") or "")
            if not path_text:
                continue
            count += 1
            path_tokens.update(_tokenize(path_text))
            semantic_tokens.update(_tokenize(str(row.get("text_preview") or "")))
            try:
                links = db.list_links(path_text)
                relationships = db.list_relationships(path_text)
            except Exception:
                links = []
                relationships = []
            link_count += len(links)
            relationship_count += len(relationships)
            for link in links:
                semantic_tokens.update(_tokenize(str(link.get("forge") or "")))
                semantic_tokens.update(_tokenize(str(link.get("relation") or "")))
                semantic_tokens.update(_tokenize(json.dumps(link.get("detail") or {}, sort_keys=True)))
            for relationship in relationships:
                semantic_tokens.update(_tokenize(str(relationship.get("dst_path") or "")))
                semantic_tokens.update(_tokenize(str(relationship.get("rel_type") or "")))
        return {
            "file_count": count,
            "link_count": link_count,
            "relationship_count": relationship_count,
            "path_tokens": path_tokens,
            "semantic_tokens": semantic_tokens,
        }
    except Exception:
        return {
            "file_count": 0,
            "link_count": 0,
            "relationship_count": 0,
            "path_tokens": set(),
            "semantic_tokens": set(),
        }


def _load_code_tokens(repo_root: Path) -> dict[str, Any]:
    candidate_roots = [
        repo_root / "data" / "code_forge",
        repo_root / "archive_forge",
        repo_root / "reports" / "code_forge_eval",
    ]
    report = {
        "link_file_count": 0,
        "registry_file_count": 0,
        "stage_counts": {},
        "latest_entries": [],
    }
    stage_counts: Counter[str] = Counter()
    latest_entries: list[dict[str, Any]] = []
    tokens: set[str] = set()
    for root in candidate_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("provenance_*.json")):
            payload = _load_json(path)
            if not payload:
                continue
            kind = "links" if path.name == "provenance_links.json" else "registry" if path.name == "provenance_registry.json" else "other"
            if kind == "links":
                report["link_file_count"] += 1
            elif kind == "registry":
                report["registry_file_count"] += 1
            stage = str(payload.get("stage") or "unknown")
            stage_counts[stage] += 1
            row = {
                "kind": kind,
                "stage": stage,
                "generated_at": payload.get("generated_at"),
                "root_path": payload.get("root_path"),
                "path": str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path),
            }
            latest_entries.append(row)
            for field in ("root_path", "path", "stage"):
                value = row.get(field)
                if isinstance(value, str):
                    tokens.update(token for token in _tokenize(value) if len(token) >= 3)
            artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                artifact_path = artifact.get("path")
                artifact_kind = artifact.get("artifact_kind")
                if isinstance(artifact_path, str):
                    tokens.update(token for token in _tokenize(artifact_path) if len(token) >= 3)
                if isinstance(artifact_kind, str):
                    tokens.update(token for token in _tokenize(artifact_kind) if len(token) >= 3)

    library_db_path = repo_root / "data" / "code_forge" / "library.sqlite"
    code_unit_count = 0
    code_file_count = 0
    if library_db_path.exists():
        try:
            with sqlite3.connect(str(library_db_path)) as conn:
                conn.row_factory = sqlite3.Row
                code_unit_count = int(conn.execute("SELECT COUNT(*) FROM code_units").fetchone()[0])
                code_file_count = int(conn.execute("SELECT COUNT(*) FROM file_records").fetchone()[0])
                rows = conn.execute(
                    """
                    SELECT file_path, qualified_name, name
                    FROM code_units
                    ORDER BY created_at DESC
                    LIMIT 12000
                    """
                ).fetchall()
                for row in rows:
                    for field in ("file_path", "qualified_name", "name"):
                        value = row[field]
                        if isinstance(value, str):
                            tokens.update(token for token in _tokenize(value) if len(token) >= 3)
                file_rows = conn.execute(
                    "SELECT file_path FROM file_records ORDER BY updated_at DESC LIMIT 6000"
                ).fetchall()
                for row in file_rows:
                    value = row[0]
                    if isinstance(value, str):
                        tokens.update(token for token in _tokenize(value) if len(token) >= 3)
        except Exception:
            code_unit_count = 0
            code_file_count = 0
    latest_entries.sort(key=lambda row: str(row.get("generated_at") or ""), reverse=True)
    report["stage_counts"] = dict(sorted(stage_counts.items()))
    report["latest_entries"] = latest_entries[:24]
    report["code_library_unit_count"] = code_unit_count
    report["code_library_file_count"] = code_file_count
    return {
        "provenance": report,
        "code_tokens": tokens,
    }


def build_bridge_audit(repo_root: Path, db_path: Path | None = None) -> dict[str, Any]:
    db_path = db_path or (repo_root / "word_forge" / "data" / "word_forge.sqlite")
    kb_path = repo_root / "data" / "kb.json"
    word_metrics = _load_word_metrics(db_path)
    knowledge_metrics = _load_knowledge_tokens(kb_path)
    code_metrics = _load_code_tokens(repo_root)
    file_metrics = _load_file_tokens(repo_root)

    base_terms = word_metrics.get("base_terms", [])
    word_terms = word_metrics.get("word_terms", set())
    knowledge_tag_tokens = knowledge_metrics.get("tag_tokens", set())
    knowledge_content_tokens = knowledge_metrics.get("content_tokens", set())
    code_tokens = code_metrics.get("code_tokens", set())
    file_tokens = set(file_metrics.get("path_tokens", set())) | set(file_metrics.get("semantic_tokens", set()))

    per_term = []
    for term in base_terms:
        word_match = term in word_terms
        knowledge_match = term in knowledge_tag_tokens or term in knowledge_content_tokens
        code_match = term in code_tokens
        file_match = term in file_tokens
        per_term.append(
            {
                "term": term,
                "word_match": word_match,
                "knowledge_match": knowledge_match,
                "code_match": code_match,
                "file_match": file_match,
                "bridge_count": int(word_match) + int(knowledge_match) + int(code_match) + int(file_match),
            }
        )
    per_term.sort(key=lambda row: (-int(row["bridge_count"]), row["term"]))

    bridge_counts = Counter(
        {
            "word": sum(1 for row in per_term if row["word_match"]),
            "knowledge": sum(1 for row in per_term if row["knowledge_match"]),
            "code": sum(1 for row in per_term if row["code_match"]),
            "file": sum(1 for row in per_term if row["file_match"]),
            "fully_bridged": sum(1 for row in per_term if row["bridge_count"] >= 4),
            "partially_bridged": sum(1 for row in per_term if row["bridge_count"] >= 2),
            "any_bridged": sum(1 for row in per_term if row["bridge_count"] >= 1),
        }
    )
    candidate_term_count = len(per_term)
    bridge_quality = {
        "candidate_term_count": candidate_term_count,
        "fully_bridged_ratio": round((bridge_counts["fully_bridged"] / candidate_term_count), 4) if candidate_term_count else 0.0,
        "partially_bridged_ratio": round((bridge_counts["partially_bridged"] / candidate_term_count), 4) if candidate_term_count else 0.0,
        "any_bridged_ratio": round((bridge_counts["any_bridged"] / candidate_term_count), 4) if candidate_term_count else 0.0,
    }

    return {
        "contract": "eidos.word_forge.bridge_audit.v1",
        "generated_at": _now_iso(),
        "repo_root": str(repo_root),
        "db_path": str(db_path),
        "kb_path": str(kb_path),
        "word_metrics": {
            "word_count": word_metrics.get("word_count", 0),
            "relationship_count": word_metrics.get("relationship_count", 0),
            "lexeme_count": word_metrics.get("lexeme_count", 0),
            "translation_count": word_metrics.get("translation_count", 0),
            "base_aligned_count": word_metrics.get("base_aligned_count", 0),
        },
        "knowledge_metrics": {
            "node_count": knowledge_metrics.get("node_count", 0),
            "tag_token_count": len(knowledge_tag_tokens),
            "content_token_count": len(knowledge_content_tokens),
        },
        "code_metrics": {
            "provenance_link_file_count": (code_metrics.get("provenance") or {}).get("link_file_count", 0),
            "provenance_registry_file_count": (code_metrics.get("provenance") or {}).get("registry_file_count", 0),
            "provenance_stage_counts": (code_metrics.get("provenance") or {}).get("stage_counts", {}),
            "code_library_unit_count": (code_metrics.get("provenance") or {}).get("code_library_unit_count", 0),
            "code_library_file_count": (code_metrics.get("provenance") or {}).get("code_library_file_count", 0),
            "code_token_count": len(code_tokens),
        },
        "file_metrics": {
            "file_count": file_metrics.get("file_count", 0),
            "link_count": file_metrics.get("link_count", 0),
            "relationship_count": file_metrics.get("relationship_count", 0),
            "path_token_count": len(file_tokens),
        },
        "bridge_counts": dict(bridge_counts),
        "bridge_quality": bridge_quality,
        "top_bridged_terms": per_term[:24],
    }


def render_bridge_audit_markdown(report: dict[str, Any]) -> str:
    word_metrics = report.get("word_metrics") or {}
    knowledge_metrics = report.get("knowledge_metrics") or {}
    code_metrics = report.get("code_metrics") or {}
    file_metrics = report.get("file_metrics") or {}
    bridge_counts = report.get("bridge_counts") or {}
    bridge_quality = report.get("bridge_quality") or {}
    rows = report.get("top_bridged_terms") or []
    lines = [
        "# Word Forge Bridge Audit",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Words: `{word_metrics.get('word_count', 0)}`",
        f"- Relationships: `{word_metrics.get('relationship_count', 0)}`",
        f"- Lexemes: `{word_metrics.get('lexeme_count', 0)}`",
        f"- Translations: `{word_metrics.get('translation_count', 0)}`",
        f"- Base aligned: `{word_metrics.get('base_aligned_count', 0)}`",
        f"- Knowledge nodes: `{knowledge_metrics.get('node_count', 0)}`",
        f"- Code provenance links: `{code_metrics.get('provenance_link_file_count', 0)}`",
        f"- Code provenance registries: `{code_metrics.get('provenance_registry_file_count', 0)}`",
        f"- Code library units: `{code_metrics.get('code_library_unit_count', 0)}`",
        f"- Code library files: `{code_metrics.get('code_library_file_count', 0)}`",
        f"- File Forge files: `{file_metrics.get('file_count', 0)}`",
        f"- File Forge links: `{file_metrics.get('link_count', 0)}`",
        f"- File Forge relationships: `{file_metrics.get('relationship_count', 0)}`",
        "",
        "## Bridge Coverage",
        "",
        f"- Word matches: `{bridge_counts.get('word', 0)}`",
        f"- Knowledge matches: `{bridge_counts.get('knowledge', 0)}`",
        f"- Code matches: `{bridge_counts.get('code', 0)}`",
        f"- File matches: `{bridge_counts.get('file', 0)}`",
        f"- Fully bridged: `{bridge_counts.get('fully_bridged', 0)}`",
        f"- Partially bridged: `{bridge_counts.get('partially_bridged', 0)}`",
        f"- Any bridged: `{bridge_counts.get('any_bridged', 0)}`",
        f"- Candidate terms: `{bridge_quality.get('candidate_term_count', 0)}`",
        f"- Full bridge ratio: `{bridge_quality.get('fully_bridged_ratio', 0.0)}`",
        f"- Partial bridge ratio: `{bridge_quality.get('partially_bridged_ratio', 0.0)}`",
        f"- Any bridge ratio: `{bridge_quality.get('any_bridged_ratio', 0.0)}`",
        "",
        "## Top Bridged Terms",
        "",
        "| Term | Word | Knowledge | Code | File | Bridge Count |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if rows:
        for row in rows:
            lines.append(
                f"| {row.get('term')} | {int(bool(row.get('word_match')))} | {int(bool(row.get('knowledge_match')))} | {int(bool(row.get('code_match')))} | {int(bool(row.get('file_match')))} | {row.get('bridge_count', 0)} |"
            )
    else:
        lines.append("| none | 0 | 0 | 0 | 0 | 0 |")
    return "\n".join(lines) + "\n"


def write_bridge_audit(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = repo_root / "reports" / "word_forge_bridge_audit"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_bridge_audit_{stamp}.json"
    md_path = report_dir / f"word_forge_bridge_audit_{stamp}.md"
    latest_json = report_dir / "latest.json"
    latest_md = report_dir / "latest.md"
    _write_json(json_path, report)
    md_path.write_text(render_bridge_audit_markdown(report), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def run_bridge_audit(repo_root: Path, db_path: Path | None = None) -> dict[str, Any]:
    runtime_dir = repo_root / "data" / "runtime"
    status_path = runtime_dir / "word_forge_bridge_audit_status.json"
    history_path = runtime_dir / "word_forge_bridge_audit_history.jsonl"
    status_payload = {
        "contract": "eidos.word_forge.bridge_audit.status.v1",
        "status": "running",
        "phase": "auditing",
        "repo_root": str(repo_root),
        "db_path": str(db_path or (repo_root / 'word_forge' / 'data' / 'word_forge.sqlite')),
        "started_at": _now_iso(),
    }
    _write_json(status_path, status_payload)
    try:
        report = build_bridge_audit(repo_root, db_path=db_path)
        artifacts = write_bridge_audit(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": _now_iso(),
                "report_path": artifacts["latest_json"],
                "fully_bridged": (report.get("bridge_counts") or {}).get("fully_bridged", 0),
            }
        )
        _write_json(status_path, status_payload)
        _append_history(history_path, status_payload)
        return {"status": status_payload, "report": report, "artifacts": artifacts}
    except Exception as exc:
        status_payload.update({
            "status": "failed",
            "phase": "failed",
            "finished_at": _now_iso(),
            "error": str(exc),
        })
        _write_json(status_path, status_payload)
        _append_history(history_path, status_payload)
        raise
