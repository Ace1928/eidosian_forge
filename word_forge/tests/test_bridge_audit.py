from __future__ import annotations

import json
from pathlib import Path

from word_forge.bridge.audit import build_bridge_audit, run_bridge_audit
from word_forge.database.database_manager import DBManager


def test_build_bridge_audit_counts_word_knowledge_and_code_matches(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBManager(db_path=db_path)
    db.insert_or_update_word("hello", "A greeting", "noun", ["hello there"])
    db.insert_or_update_lexeme("bonjour", "fr", base_term="hello", gloss="hello", source="test")
    lexeme_id = db.get_lexeme_id("bonjour", "fr")
    db.add_translation(lexeme_id, "en", "hello", source="test")

    kb_path = repo_root / "data" / "kb.json"
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    kb_path.write_text(
        json.dumps(
            {
                "nodes": {
                    "n1": {
                        "id": "n1",
                        "content": "hello systems integration",
                        "metadata": {"tags": ["hello", "greeting"]},
                        "links": [],
                    }
                },
                "concept_map": {},
            }
        ),
        encoding="utf-8",
    )

    from file_forge.core import FileForge

    forge = FileForge(base_path=repo_root)
    archive_file = repo_root / "archive_forge" / "hello_archive.txt"
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    archive_file.write_text("hello archive", encoding="utf-8")
    forge.index_directory(repo_root / "archive_forge", db_path=repo_root / "data" / "file_forge" / "library.sqlite")

    prov_path = repo_root / "data" / "code_forge" / "run1" / "provenance_links.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-28T00:00:00Z",
                "stage": "hello_stage",
                "root_path": "/tmp/hello_repo",
            }
        ),
        encoding="utf-8",
    )

    report = build_bridge_audit(repo_root=repo_root, db_path=db_path)

    assert report["word_metrics"]["base_aligned_count"] == 1
    assert report["bridge_counts"]["word"] == 1
    assert report["bridge_counts"]["knowledge"] == 1
    assert report["bridge_counts"]["code"] == 1
    assert report["bridge_counts"]["file"] == 1
    assert report["bridge_counts"]["fully_bridged"] == 1
    assert report["bridge_counts"]["partially_bridged"] == 1
    assert report["bridge_counts"]["any_bridged"] == 1
    assert report["bridge_quality"]["candidate_term_count"] == 1
    assert report["bridge_quality"]["fully_bridged_ratio"] == 1.0


def test_run_bridge_audit_writes_runtime_and_latest_report(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.insert_or_update_word("atlas", "A support system", "noun", ["atlas route"])
    db.insert_or_update_lexeme("atlas", "en", base_term="atlas", source="test")
    (repo_root / "data").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "kb.json").write_text(json.dumps({"nodes": {}, "concept_map": {}}), encoding="utf-8")

    result = run_bridge_audit(repo_root=repo_root, db_path=db_path)

    assert result["status"]["status"] == "completed"
    assert (repo_root / "data" / "runtime" / "word_forge_bridge_audit_status.json").exists()
    assert (repo_root / "data" / "runtime" / "word_forge_bridge_audit_history.jsonl").exists()
    assert Path(result["artifacts"]["latest_json"]).exists()
    assert Path(result["artifacts"]["latest_markdown"]).exists()


def test_build_bridge_audit_uses_fileforge_semantic_tokens_and_code_library(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBManager(db_path=db_path)
    db.insert_or_update_word("pipeline", "A connected flow", "noun", ["pipeline route"])
    db.insert_or_update_lexeme("pipeline", "en", base_term="pipeline", source="test")

    kb_path = repo_root / "data" / "kb.json"
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    kb_path.write_text(json.dumps({"nodes": {}, "concept_map": {}}), encoding="utf-8")

    from file_forge.library import FileLibraryDB

    file_db = FileLibraryDB(repo_root / "data" / "file_forge" / "library.sqlite")
    file_db.upsert_file(
        file_path=repo_root / "notes.txt",
        payload=b"semantic notes",
        size_bytes=len(b"semantic notes"),
        modified_ns=1,
        kind="document",
        mime_type="text/plain",
        encoding="utf-8",
        text_preview="pipeline orchestration notes",
        links=[{"forge": "knowledge_forge", "relation": "documents", "detail": {"topic": "pipeline"}}],
    )

    import sqlite3

    code_db_path = repo_root / "data" / "code_forge" / "library.sqlite"
    code_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(code_db_path)) as conn:
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
            CREATE TABLE file_records (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT,
                analysis_version INTEGER,
                updated_at TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO code_units (id, file_path, qualified_name, name, language, unit_type, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("u1", str(repo_root / "pipeline_runner.py"), "pipeline.runner.main", "main", "python", "function", "2026-03-28T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO file_records (file_path, content_hash, analysis_version, updated_at) VALUES (?, ?, ?, ?)",
            (str(repo_root / "pipeline_runner.py"), "abc", 1, "2026-03-28T00:00:00Z"),
        )

    report = build_bridge_audit(repo_root=repo_root, db_path=db_path)

    assert report["bridge_counts"]["code"] == 1
    assert report["bridge_counts"]["file"] == 1
    assert report["code_metrics"]["code_library_unit_count"] == 1
    assert report["file_metrics"]["link_count"] == 1
