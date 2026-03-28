from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    current: list[str] = []
    for ch in (text or "").lower():
        if ch.isalnum() or ch in {"_", "-"}:
            current.append(ch)
            continue
        if current:
            token = "".join(current)
            tokens.add(token)
            for part in token.replace("-", "_").split("_"):
                if part:
                    tokens.add(part)
            current = []
    if current:
        token = "".join(current)
        tokens.add(token)
        for part in token.replace("-", "_").split("_"):
            if part:
                tokens.add(part)
    return {token for token in tokens if len(token) >= 3}


def _safe_relpath(path_text: str, forge_root: Path) -> str:
    path = Path(path_text)
    try:
        return str(path.relative_to(forge_root))
    except Exception:
        return str(path)


def _word_rows(conn: sqlite3.Connection, *, limit: int = 220) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT w.term, w.part_of_speech,
               AVG(er.valence) as avg_valence,
               AVG(er.arousal) as avg_arousal
        FROM words w
        LEFT JOIN emotional_relationships er ON w.id = er.word_id
        GROUP BY w.id
        ORDER BY w.last_refreshed DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()


def _relationship_rows(conn: sqlite3.Connection, *, limit: int = 400) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT w.term AS term1, r.related_term AS term2, r.relationship_type
        FROM relationships r
        JOIN words w ON w.id = r.word_id
        ORDER BY r.id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()


def _lexeme_rows(conn: sqlite3.Connection, *, limit: int = 180) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, lemma, lang, part_of_speech, gloss, base_term FROM lexemes ORDER BY last_refreshed DESC LIMIT ?",
        (int(limit),),
    ).fetchall()


def _translation_rows(conn: sqlite3.Connection, *, limit: int = 260) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT l.lemma, l.lang, l.base_term, t.target_lang, t.target_term, t.relation
        FROM translations t
        JOIN lexemes l ON l.id = t.lexeme_id
        ORDER BY t.last_refreshed DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()


def build_word_graph_payload(
    *,
    db_path: Path,
    forge_root: Path,
    kb_payload: Optional[dict[str, Any]] = None,
    code_report: Optional[dict[str, Any]] = None,
    file_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not db_path.exists():
        return {"nodes": [], "edges": [], "error": f"Database not found: {db_path}"}

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        words = _word_rows(conn)
        rels = _relationship_rows(conn)
        lexemes = _lexeme_rows(conn)
        translations = _translation_rows(conn)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_ids: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    bridge_terms: set[str] = set()

    def add_node(node: Dict[str, Any]) -> None:
        node_id = str(node.get("id") or "")
        if not node_id or node_id in node_ids:
            return
        node_ids.add(node_id)
        nodes.append(node)

    def add_edge(src: str, dst: str, label: str) -> None:
        edge = tuple(sorted((src, dst)))
        if edge in seen_edges:
            return
        seen_edges.add(edge)
        edges.append({"from": src, "to": dst, "label": label})

    for row in words:
        term = str(row["term"])
        bridge_terms.add(term.lower())
        valence = row["avg_valence"] or 0.0
        arousal = row["avg_arousal"] or 0.5
        red = int(255 * (1 - (valence + 1) / 2))
        green = int(255 * ((valence + 1) / 2))
        color = f"rgb({red}, {green}, 150)"
        add_node(
            {
                "id": f"word:{term}",
                "label": term,
                "title": f"<b>Word</b><br>{term} ({row['part_of_speech']})<br>Valence: {valence:.2f}<br>Arousal: {arousal:.2f}",
                "group": "lexicon",
                "value": 10 + (arousal * 20),
                "color": {"background": color, "border": "#edf7ff"},
                "metadata": dict(row),
            }
        )

    for row in lexemes:
        lemma = str(row["lemma"])
        lang = str(row["lang"])
        base_term = str(row["base_term"] or "")
        bridge_terms.add(lemma.lower())
        if base_term:
            bridge_terms.add(base_term.lower())
        lexeme_id = f"lexeme:{lang}:{lemma}"
        add_node(
            {
                "id": lexeme_id,
                "label": f"{lemma} [{lang}]",
                "title": f"<b>Lexeme</b><br>{lemma} ({lang})<br>{row['gloss'] or ''}",
                "group": "multilingual",
                "value": 12,
                "color": {"background": "#60a5fa", "border": "#bfdbfe"},
                "metadata": dict(row),
            }
        )
        if base_term:
            word_id = f"word:{base_term}"
            if word_id not in node_ids:
                add_node(
                    {
                        "id": word_id,
                        "label": base_term,
                        "title": f"<b>Word</b><br>{base_term}",
                        "group": "lexicon",
                        "value": 12,
                        "color": {"background": "#fbbf24", "border": "#edf7ff"},
                        "metadata": {"term": base_term, "inferred": True},
                    }
                )
            add_edge(lexeme_id, word_id, "base_alignment")

    for row in rels:
        src = f"word:{row['term1']}"
        dst = f"word:{row['term2']}"
        if src in node_ids and dst in node_ids:
            add_edge(src, dst, str(row["relationship_type"]))

    for row in translations:
        source_id = f"lexeme:{row['lang']}:{row['lemma']}"
        target_id = f"translation:{row['target_lang']}:{row['target_term']}"
        if source_id not in node_ids:
            continue
        add_node(
            {
                "id": target_id,
                "label": f"{row['target_term']} [{row['target_lang']}]",
                "title": f"<b>Translation</b><br>{row['target_term']} ({row['target_lang']})",
                "group": "translation",
                "value": 10,
                "color": {"background": "#34d399", "border": "#a7f3d0"},
                "metadata": dict(row),
            }
        )
        add_edge(source_id, target_id, str(row["relation"]))

    kb_nodes = kb_payload.get("nodes") if isinstance((kb_payload or {}).get("nodes"), dict) else {}
    matched_terms = 0
    for kb_id, kb_node in list(kb_nodes.items())[:500]:
        if matched_terms >= 40:
            break
        if not isinstance(kb_node, dict):
            continue
        metadata = kb_node.get("metadata") if isinstance(kb_node.get("metadata"), dict) else {}
        tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
        lowered_tags = {str(tag).strip().lower() for tag in tags if isinstance(tag, str)}
        shared_terms = sorted(term for term in bridge_terms if term in lowered_tags)
        if not shared_terms:
            continue
        kb_node_id = f"kb:{kb_id}"
        add_node(
            {
                "id": kb_node_id,
                "label": kb_id[:8],
                "title": f"<b>Knowledge</b><br>{str(kb_node.get('content') or '')[:200]}",
                "group": "knowledge",
                "value": 14,
                "color": {"background": "#4fd1c5", "border": "#7dd3fc"},
                "metadata": kb_node,
            }
        )
        for term in shared_terms[:3]:
            src = f"word:{term}"
            if src in node_ids:
                add_edge(src, kb_node_id, "knowledge_tag")
        matched_terms += 1

    latest_entries = (code_report or {}).get("latest_entries") or []
    for index, entry in enumerate(latest_entries[:20]):
        path_text = str(entry.get("path") or entry.get("root_path") or f"code_entry_{index}")
        stage = str(entry.get("stage") or "code")
        code_id = f"code:{index}:{path_text}"
        token_source = f"{path_text}\n{stage}"
        shared_terms = sorted(term for term in bridge_terms if term in _tokenize(token_source))
        if not shared_terms:
            continue
        add_node(
            {
                "id": code_id,
                "label": Path(path_text).name or stage,
                "title": f"<b>Code Provenance</b><br>{path_text}<br>Stage: {stage}",
                "group": "code",
                "value": 11,
                "color": {"background": "#c084fc", "border": "#e9d5ff"},
                "metadata": dict(entry),
            }
        )
        for term in shared_terms[:2]:
            src = f"word:{term}"
            if src in node_ids:
                add_edge(src, code_id, "code_provenance")

    recent_files = (file_summary or {}).get("recent_files") or []
    for entry in recent_files[:16]:
        path_text = str(entry.get("file_path") or "")
        if not path_text:
            continue
        shared_terms = sorted(term for term in bridge_terms if term in _tokenize(path_text))
        if not shared_terms:
            continue
        file_id = f"file:{path_text}"
        add_node(
            {
                "id": file_id,
                "label": Path(path_text).name,
                "title": f"<b>File Forge</b><br>{path_text}<br>Kind: {entry.get('kind', 'unknown')}",
                "group": "file",
                "value": 10,
                "color": {"background": "#94a3b8", "border": "#edf7ff"},
                "metadata": dict(entry),
            }
        )
        for term in shared_terms[:2]:
            src = f"word:{term}"
            if src in node_ids:
                add_edge(src, file_id, "file_path")

    return {"nodes": nodes, "edges": edges}
