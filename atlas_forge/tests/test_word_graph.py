from __future__ import annotations

import sqlite3
from pathlib import Path

from atlas_forge.word_graph import build_word_graph_payload


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


def test_build_word_graph_payload_includes_code_file_and_knowledge_bridges(tmp_path: Path) -> None:
    db_path = tmp_path / "word_forge.sqlite"
    _make_db(db_path)
    payload = build_word_graph_payload(
        db_path=db_path,
        forge_root=tmp_path,
        kb_payload={
            "nodes": {
                "k1": {
                    "id": "k1",
                    "content": "archive lifecycle coverage",
                    "metadata": {"tags": ["archive", "retirement"]},
                    "links": [],
                }
            }
        },
        code_report={
            "latest_entries": [
                {"path": "archive_forge/archive_module.py", "stage": "archive_digester", "generated_at": "2026-03-28T00:00:00Z"}
            ]
        },
        file_summary={
            "recent_files": [
                {"file_path": str(tmp_path / "archive_notes.md"), "kind": "document", "updated_at": "2026-03-28T00:00:00Z"}
            ]
        },
    )

    groups = {node["group"] for node in payload["nodes"]}
    labels = {edge["label"] for edge in payload["edges"]}

    assert "lexicon" in groups
    assert "multilingual" in groups
    assert "translation" in groups
    assert "knowledge" in groups
    assert "code" in groups
    assert "file" in groups
    assert "knowledge_tag" in labels
    assert "code_provenance" in labels
    assert "file_path" in labels
