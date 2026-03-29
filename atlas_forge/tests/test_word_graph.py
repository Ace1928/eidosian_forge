from __future__ import annotations

import sqlite3
from pathlib import Path

from atlas_forge.word_graph import (
    build_word_graph_neighbor_payload,
    build_word_graph_payload,
    summarize_word_graph_communities,
)


def _make_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE words (
                id INTEGER PRIMARY KEY,
                term TEXT NOT NULL,
                definition TEXT,
                part_of_speech TEXT,
                usage_examples TEXT,
                language TEXT DEFAULT 'en',
                last_refreshed REAL DEFAULT 0
            );
            CREATE TABLE emotional_relationships (
                id INTEGER PRIMARY KEY,
                word_id INTEGER,
                related_term TEXT,
                relationship_type TEXT,
                valence REAL,
                arousal REAL,
                last_updated REAL
            );
            CREATE TABLE relationships (
                id INTEGER PRIMARY KEY,
                word_id INTEGER NOT NULL,
                related_term TEXT NOT NULL,
                relationship_type TEXT NOT NULL
            );
            CREATE TABLE lexemes (
                id INTEGER PRIMARY KEY,
                lemma TEXT NOT NULL,
                lang TEXT NOT NULL,
                part_of_speech TEXT,
                gloss TEXT,
                base_term TEXT,
                source TEXT,
                last_refreshed REAL NOT NULL
            );
            CREATE TABLE translations (
                id INTEGER PRIMARY KEY,
                lexeme_id INTEGER NOT NULL,
                target_lang TEXT NOT NULL,
                target_term TEXT NOT NULL,
                relation TEXT NOT NULL,
                source TEXT,
                last_refreshed REAL NOT NULL
            );
            CREATE TABLE morphemes (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                type TEXT,
                meaning TEXT,
                last_updated REAL
            );
            CREATE TABLE word_morphemes (
                word_id INTEGER NOT NULL,
                morpheme_id INTEGER NOT NULL,
                position INTEGER NOT NULL
            );
            CREATE TABLE lexeme_morphemes (
                lexeme_id INTEGER NOT NULL,
                morpheme_id INTEGER NOT NULL,
                position INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO words (id, term, definition, part_of_speech, usage_examples, language, last_refreshed) VALUES (1, 'archive', 'desc', 'n', '[]', 'en', 1)"
        )
        conn.execute(
            "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (1, 'repository', 'synonym')"
        )
        conn.execute(
            "INSERT INTO lexemes (id, lemma, lang, part_of_speech, gloss, base_term, source, last_refreshed) VALUES (1, 'archivo', 'es', 'n', 'archive', 'archive', 'test', 1)"
        )
        conn.execute(
            "INSERT INTO translations (lexeme_id, target_lang, target_term, relation, source, last_refreshed) VALUES (1, 'en', 'archive', 'translation', 'test', 1)"
        )
        conn.execute("INSERT INTO morphemes (id, text, type, meaning, last_updated) VALUES (1, 'arch', 'root', 'archive', 1)")
        conn.execute("INSERT INTO morphemes (id, text, type, meaning, last_updated) VALUES (2, 'ive', 'suffix', 'state', 1)")
        conn.execute("INSERT INTO word_morphemes (word_id, morpheme_id, position) VALUES (1, 1, 0)")
        conn.execute("INSERT INTO word_morphemes (word_id, morpheme_id, position) VALUES (1, 2, 1)")
        conn.execute("INSERT INTO lexeme_morphemes (lexeme_id, morpheme_id, position) VALUES (1, 1, 0)")
        conn.execute("INSERT INTO lexeme_morphemes (lexeme_id, morpheme_id, position) VALUES (1, 2, 1)")


def _sample_inputs(tmp_path: Path) -> dict[str, object]:
    return {
        "kb_payload": {
            "nodes": {
                "k1": {
                    "id": "k1",
                    "content": "archive lifecycle coverage",
                    "metadata": {"tags": ["archive", "retirement"]},
                    "links": [],
                }
            }
        },
        "code_report": {
            "latest_entries": [
                {"path": "archive_forge/archive_module.py", "stage": "archive_digester", "generated_at": "2026-03-28T00:00:00Z"}
            ]
        },
        "file_summary": {
            "recent_files": [
                {"file_path": str(tmp_path / "archive_notes.md"), "kind": "document", "updated_at": "2026-03-28T00:00:00Z"}
            ]
        },
    }


def test_build_word_graph_payload_includes_code_file_and_knowledge_bridges(tmp_path: Path) -> None:
    db_path = tmp_path / "word_forge.sqlite"
    _make_db(db_path)
    payload = build_word_graph_payload(
        db_path=db_path,
        forge_root=tmp_path,
        **_sample_inputs(tmp_path),
    )

    groups = {node["group"] for node in payload["nodes"]}
    labels = {edge["label"] for edge in payload["edges"]}

    assert "lexicon" in groups
    assert "multilingual" in groups
    assert "translation" in groups
    assert "knowledge" in groups
    assert "code" in groups
    assert "file" in groups
    assert "morpheme" in groups
    assert "knowledge_tag" in labels
    assert "code_provenance" in labels
    assert "file_path" in labels


def test_build_word_graph_neighbor_payload_expands_code_and_file_nodes(tmp_path: Path) -> None:
    db_path = tmp_path / "word_forge.sqlite"
    _make_db(db_path)
    sample = _sample_inputs(tmp_path)

    code_neighbors = build_word_graph_neighbor_payload(
        node_id="code:0:archive_forge/archive_module.py",
        db_path=db_path,
        forge_root=tmp_path,
        **sample,
    )
    code_ids = {node["id"] for node in code_neighbors["nodes"]}
    assert "word:archive" in code_ids
    assert "morpheme:arch" in code_ids
    assert "kb:k1" in code_ids

    file_node_id = f"file:{tmp_path / 'archive_notes.md'}"
    file_neighbors = build_word_graph_neighbor_payload(
        node_id=file_node_id,
        db_path=db_path,
        forge_root=tmp_path,
        **sample,
    )
    file_ids = {node["id"] for node in file_neighbors["nodes"]}
    assert "word:archive" in file_ids
    assert "morpheme:arch" in file_ids
    assert "code:0:archive_forge/archive_module.py" in file_ids


def test_summarize_word_graph_communities_detects_multi_layer_anchor_terms(tmp_path: Path) -> None:
    db_path = tmp_path / "word_forge.sqlite"
    _make_db(db_path)
    payload = build_word_graph_payload(
        db_path=db_path,
        forge_root=tmp_path,
        **_sample_inputs(tmp_path),
    )

    summary = summarize_word_graph_communities(payload, limit=8)

    assert summary["contract"] == "eidos.atlas.word_graph.communities.v1"
    assert summary["community_count"] >= 1
    top = summary["top_communities"][0]
    assert top["anchor_term"] == "archive"
    assert top["layer_count"] >= 4
    assert "morpheme" in top["layers"]
    assert "code" in top["layers"]
    assert "file" in top["layers"]
    assert "knowledge" in top["layers"]


def test_build_word_graph_payload_uses_code_library_and_file_preview_fallbacks(tmp_path: Path) -> None:
    db_path = tmp_path / "word_forge.sqlite"
    _make_db(db_path)

    code_db = tmp_path / "data" / "code_forge" / "library.sqlite"
    code_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(code_db)) as conn:
        conn.executescript(
            """
            CREATE TABLE code_units (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                qualified_name TEXT,
                name TEXT,
                language TEXT,
                unit_type TEXT,
                created_at TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO code_units (id, file_path, qualified_name, name, language, unit_type, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("u1", str(tmp_path / "archive_pipeline.py"), "archive.pipeline.run", "run", "python", "function", "2026-03-28T00:00:00Z"),
        )

    payload = build_word_graph_payload(
        db_path=db_path,
        forge_root=tmp_path,
        kb_payload={"nodes": {}},
        code_report={"latest_entries": []},
        file_summary={
            "recent_files": [
                {
                    "file_path": str(tmp_path / "notes.md"),
                    "kind": "document",
                    "text_preview": "archive pipeline notes",
                    "links": [{"forge": "knowledge_forge", "relation": "documents", "detail": {"topic": "archive"}}],
                }
            ]
        },
    )

    groups = {node["group"] for node in payload["nodes"]}
    labels = {edge["label"] for edge in payload["edges"]}
    assert "code" in groups
    assert "file" in groups
    assert "code_provenance" in labels
    assert "file_path" in labels
