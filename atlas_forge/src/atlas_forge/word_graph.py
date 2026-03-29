from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _morpheme_rows(conn: sqlite3.Connection, *, limit: int = 320) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT COALESCE(NULLIF(l.base_term, ''), l.lemma) AS anchor_term,
               l.lemma,
               l.lang,
               m.text AS morpheme_text,
               lm.position,
               'lexeme' AS source_type
        FROM lexemes l
        JOIN lexeme_morphemes lm ON lm.lexeme_id = l.id
        JOIN morphemes m ON m.id = lm.morpheme_id
        UNION ALL
        SELECT w.term AS anchor_term,
               w.term AS lemma,
               '' AS lang,
               m.text AS morpheme_text,
               wm.position,
               'word' AS source_type
        FROM words w
        JOIN word_morphemes wm ON wm.word_id = w.id
        JOIN morphemes m ON m.id = wm.morpheme_id
        ORDER BY anchor_term ASC, source_type DESC, position ASC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()


def _coerce_code_entries(code_report: Optional[dict[str, Any]], forge_root: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    latest_entries = (code_report or {}).get("latest_entries") or []
    coerced: list[dict[str, Any]] = []
    if isinstance(latest_entries, list):
        for entry in latest_entries:
            if isinstance(entry, dict):
                coerced.append(entry)
                if len(coerced) >= limit:
                    return coerced[:limit]

    for path in sorted((forge_root / "data" / "code_forge").rglob("provenance_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        base_entry = {
            "path": str(payload.get("path") or payload.get("root_path") or path.parent),
            "root_path": str(payload.get("root_path") or ""),
            "stage": str(payload.get("stage") or "code"),
            "generated_at": str(payload.get("generated_at") or ""),
        }
        coerced.append(base_entry)
        artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            artifact_path = str(artifact.get("path") or "")
            if not artifact_path:
                continue
            coerced.append(
                {
                    "path": artifact_path,
                    "root_path": str(payload.get("root_path") or ""),
                    "stage": str(artifact.get("artifact_kind") or payload.get("stage") or "code"),
                    "generated_at": str(payload.get("generated_at") or ""),
                }
            )
            if len(coerced) >= limit:
                return coerced[:limit]
        if len(coerced) >= limit:
            return coerced[:limit]

    library_db = forge_root / "data" / "code_forge" / "library.sqlite"
    if library_db.exists():
        try:
            with sqlite3.connect(str(library_db)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT file_path, qualified_name, name, language, unit_type, created_at
                    FROM code_units
                    ORDER BY created_at DESC
                    LIMIT 300
                    """
                ).fetchall()
            for row in rows:
                path_text = str(row["file_path"] or "")
                if not path_text:
                    continue
                coerced.append(
                    {
                        "path": path_text,
                        "root_path": str(forge_root),
                        "stage": str(row["unit_type"] or row["language"] or "code_library"),
                        "qualified_name": str(row["qualified_name"] or ""),
                        "name": str(row["name"] or ""),
                        "generated_at": str(row["created_at"] or ""),
                    }
                )
                if len(coerced) >= limit:
                    return coerced[:limit]
        except Exception:
            pass
    return coerced[:limit]


def _coerce_file_entries(
    file_summary: Optional[dict[str, Any]],
    forge_root: Path,
    *,
    bridge_terms: set[str],
    limit: int = 32,
) -> list[dict[str, Any]]:
    recent_files = (file_summary or {}).get("recent_files") or []
    entries: list[dict[str, Any]] = [entry for entry in recent_files if isinstance(entry, dict)]
    if len(entries) >= limit:
        return entries[:limit]

    db_path_text = str((file_summary or {}).get("db_path") or (forge_root / "data" / "file_forge" / "library.db"))
    try:
        from file_forge.library import FileLibraryDB  # type: ignore

        db = FileLibraryDB(Path(db_path_text))
        existing_paths = {str(entry.get("file_path") or "") for entry in entries}
        matched: list[dict[str, Any]] = []
        for row in db.iter_file_records(limit=5000):
            if not isinstance(row, dict):
                continue
            file_path = str(row.get("file_path") or "")
            if not file_path or file_path in existing_paths:
                continue
            if bridge_terms and not (bridge_terms & _tokenize(file_path)):
                continue
            matched.append(
                {
                    "file_path": file_path,
                    "kind": str(row.get("kind") or "unknown"),
                    "updated_at": row.get("updated_at") or row.get("modified_ns") or "",
                    "size_bytes": int(row.get("size_bytes") or 0),
                    "content_hash": row.get("content_hash"),
                    "text_preview": str(row.get("text_preview") or ""),
                    "links": db.list_links(file_path),
                    "relationships": db.list_relationships(file_path),
                }
            )
            if len(entries) + len(matched) >= limit:
                break
        entries.extend(matched)
    except Exception:
        pass
    return entries[:limit]


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
        morphemes = _morpheme_rows(conn)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_ids: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()
    bridge_terms: set[str] = set()
    term_targets: dict[str, set[str]] = {}

    def add_node(node: Dict[str, Any]) -> None:
        node_id = str(node.get("id") or "")
        if not node_id or node_id in node_ids:
            return
        node_ids.add(node_id)
        nodes.append(node)

    def add_edge(src: str, dst: str, label: str) -> None:
        edge = (src, dst, label)
        reverse = (dst, src, label)
        if edge in seen_edges or reverse in seen_edges:
            return
        seen_edges.add(edge)
        edges.append({"from": src, "to": dst, "label": label})

    def register_term_node(term: str, node_id: str) -> None:
        normalized = str(term or "").strip().lower()
        if not normalized:
            return
        term_targets.setdefault(normalized, set()).add(node_id)
        bridge_terms.add(normalized)

    for row in words:
        term = str(row["term"])
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
        register_term_node(term, f"word:{term}")

    for row in lexemes:
        lemma = str(row["lemma"])
        lang = str(row["lang"])
        base_term = str(row["base_term"] or "")
        if base_term:
            register_term_node(base_term, f"word:{base_term}")
        lexeme_id = f"lexeme:{lang}:{lemma}"
        add_node(
            {
                "id": lexeme_id,
                "label": f"{lemma} [{lang}]",
                "title": f"<b>Lexeme</b><br>{lemma} ({lang})<br>{row['gloss'] or ''}",
                "group": "multilingual",
                "value": 11,
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
            register_term_node(base_term, word_id)
            add_edge(lexeme_id, word_id, "base_alignment")

    for row in morphemes:
        morpheme_text = str(row["morpheme_text"] or "").strip().lower()
        if not morpheme_text:
            continue
        morpheme_id = f"morpheme:{morpheme_text}"
        add_node(
            {
                "id": morpheme_id,
                "label": morpheme_text,
                "title": f"<b>Morpheme</b><br>{morpheme_text}",
                "group": "morpheme",
                "value": 8,
                "color": {"background": "#fca5a5", "border": "#fecaca"},
                "metadata": dict(row),
            }
        )
        register_term_node(morpheme_text, morpheme_id)
        anchor_term = str(row["anchor_term"] or "").strip().lower()
        lang = str(row["lang"] or "").strip()
        lemma = str(row["lemma"] or "").strip()
        if lang and lemma:
            lexeme_id = f"lexeme:{lang}:{lemma}"
            if lexeme_id in node_ids:
                add_edge(lexeme_id, morpheme_id, "has_morpheme")
        word_id = f"word:{anchor_term}"
        if anchor_term and word_id in node_ids:
            add_edge(word_id, morpheme_id, "has_morpheme")


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
            for src in sorted(term_targets.get(term, set()))[:2]:
                if src in node_ids:
                    add_edge(src, kb_node_id, "knowledge_tag")
        matched_terms += 1

    latest_entries = _coerce_code_entries(code_report, forge_root, limit=40)
    for index, entry in enumerate(latest_entries[:20]):
        path_text = str(entry.get("path") or entry.get("root_path") or f"code_entry_{index}")
        stage = str(entry.get("stage") or "code")
        code_id = f"code:{index}:{path_text}"
        token_source = f"{path_text}\n{stage}\n{entry.get('qualified_name') or ''}\n{entry.get('name') or ''}"
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
            for src in sorted(term_targets.get(term, set()))[:2]:
                if src in node_ids:
                    add_edge(src, code_id, "code_provenance")

    recent_files = _coerce_file_entries(file_summary, forge_root, bridge_terms=bridge_terms, limit=48)
    for entry in recent_files[:48]:
        path_text = str(entry.get("file_path") or "")
        if not path_text:
            continue
        token_source = [path_text, str(entry.get("text_preview") or "")]
        for link in entry.get("links") or []:
            if isinstance(link, dict):
                token_source.append(str(link.get("forge") or ""))
                token_source.append(str(link.get("relation") or ""))
                token_source.append(json.dumps(link.get("detail") or {}, sort_keys=True))
        for relationship in entry.get("relationships") or []:
            if isinstance(relationship, dict):
                token_source.append(str(relationship.get("dst_path") or ""))
                token_source.append(str(relationship.get("rel_type") or ""))
        shared_terms = sorted(term for term in bridge_terms if term in _tokenize("\n".join(token_source)))
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
            for src in sorted(term_targets.get(term, set()))[:2]:
                if src in node_ids:
                    add_edge(src, file_id, "file_path")

    return {"nodes": nodes, "edges": edges}


def build_word_graph_neighbor_payload(
    *,
    node_id: str,
    db_path: Path,
    forge_root: Path,
    kb_payload: Optional[dict[str, Any]] = None,
    code_report: Optional[dict[str, Any]] = None,
    file_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = build_word_graph_payload(
        db_path=db_path,
        forge_root=forge_root,
        kb_payload=kb_payload,
        code_report=code_report,
        file_summary=file_summary,
    )
    nodes_by_id = {str(node.get("id")): node for node in payload.get("nodes", []) if isinstance(node, dict)}
    if node_id not in nodes_by_id:
        return {"nodes": [], "edges": []}

    relevant_edges: list[dict[str, Any]] = []
    relevant_ids: set[str] = {node_id}
    seen_edge_keys: set[tuple[str, str, str]] = set()

    def collect_edge(edge: dict[str, Any]) -> None:
        src = str(edge.get("from") or "")
        dst = str(edge.get("to") or "")
        label = str(edge.get("label") or "")
        if not src or not dst:
            return
        key = (src, dst, label)
        if key in seen_edge_keys:
            return
        seen_edge_keys.add(key)
        relevant_edges.append(edge)
        relevant_ids.add(src)
        relevant_ids.add(dst)

    immediate_neighbors: set[str] = set()
    for edge in payload.get("edges", []):
        if edge.get("from") == node_id or edge.get("to") == node_id:
            collect_edge(edge)
            immediate_neighbors.add(str(edge.get("to") if edge.get("from") == node_id else edge.get("from")))

    if node_id.startswith(("code:", "file:", "kb:", "morpheme:")):
        for edge in payload.get("edges", []):
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            other = None
            if src in immediate_neighbors and dst != node_id:
                other = dst
            elif dst in immediate_neighbors and src != node_id:
                other = src
            if not other:
                continue
            if other.startswith(("word:", "lexeme:", "translation:", "kb:", "code:", "file:", "morpheme:")):
                collect_edge(edge)

    return {
        "nodes": [nodes_by_id[item] for item in sorted(relevant_ids) if item in nodes_by_id],
        "edges": relevant_edges,
    }


def summarize_word_graph_communities(payload: dict[str, Any], *, limit: int = 12) -> dict[str, Any]:
    nodes = [node for node in payload.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in payload.get("edges", []) if isinstance(edge, dict)]
    nodes_by_id = {str(node.get("id") or ""): node for node in nodes if node.get("id")}

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes_by_id}
    for edge in edges:
        src = str(edge.get("from") or "")
        dst = str(edge.get("to") or "")
        if src in adjacency and dst in adjacency:
            adjacency[src].add(dst)
            adjacency[dst].add(src)

    layer_totals: Counter[str] = Counter()
    communities: list[dict[str, Any]] = []
    layer_order = ["knowledge", "code", "file", "morpheme", "multilingual", "translation"]

    for node_id, node in nodes_by_id.items():
        if not node_id.startswith("word:"):
            continue
        neighbors = sorted(adjacency.get(node_id) or set())
        layer_members: dict[str, list[str]] = {layer: [] for layer in layer_order}
        for neighbor_id in neighbors:
            neighbor = nodes_by_id.get(neighbor_id) or {}
            group = str(neighbor.get("group") or "")
            if group in layer_members:
                layer_members[group].append(neighbor_id)
        active_layers = [layer for layer in layer_order if layer_members[layer]]
        if len(active_layers) < 2:
            continue
        for layer in active_layers:
            layer_totals[layer] += 1
        communities.append(
            {
                "community_id": f"word-community:{node_id.split(':', 1)[1]}",
                "anchor_node": node_id,
                "anchor_term": str(node.get("label") or node_id.split(":", 1)[1]),
                "layer_count": len(active_layers),
                "neighbor_count": len(neighbors),
                "layers": active_layers,
                "knowledge_nodes": len(layer_members["knowledge"]),
                "code_nodes": len(layer_members["code"]),
                "file_nodes": len(layer_members["file"]),
                "morpheme_nodes": len(layer_members["morpheme"]),
                "multilingual_nodes": len(layer_members["multilingual"]),
                "translation_nodes": len(layer_members["translation"]),
                "members": {layer: layer_members[layer][:8] for layer in active_layers},
            }
        )

    communities.sort(
        key=lambda row: (
            -int(row.get("layer_count") or 0),
            -int(row.get("neighbor_count") or 0),
            str(row.get("anchor_term") or ""),
        )
    )
    top_communities = communities[: max(1, int(limit))]
    return {
        "contract": "eidos.atlas.word_graph.communities.v1",
        "community_count": len(communities),
        "layer_totals": dict(layer_totals),
        "top_communities": top_communities,
    }
