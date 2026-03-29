from __future__ import annotations

import sqlite3
from pathlib import Path

from word_forge.database.database_manager import DBManager
from word_forge.multilingual.polyglot_runtime import run_polyglot_decomposition


def test_run_polyglot_decomposition_creates_lexeme_morphemes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.insert_or_update_lexeme("integracion", "es", base_term="integration", source="seed")
    db.insert_or_update_lexeme("canalizacion", "es", base_term="pipeline", source="seed")

    result = run_polyglot_decomposition(repo_root=repo_root, db_path=db_path, lang="es", force=True)

    assert result["status"]["status"] == "completed"
    report = result["report"]
    assert report is not None
    assert report["processed_lexemes"] == 2
    assert report["after"]["decomposed_lexeme_count"] >= 2
    assert report["after"]["lexeme_morpheme_count"] >= 4
    assert any(len(row.get("morphemes") or []) >= 2 for row in report["samples"])

    with sqlite3.connect(str(db_path)) as conn:
        count = int(conn.execute("SELECT COUNT(*) FROM lexeme_morphemes").fetchone()[0])
    assert count >= 4


def test_run_polyglot_decomposition_skips_when_unchanged(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.insert_or_update_lexeme("archivo", "es", base_term="archive", source="seed")

    first = run_polyglot_decomposition(repo_root=repo_root, db_path=db_path, lang="es", limit=1, force=True)
    assert first["status"]["status"] == "completed"

    second = run_polyglot_decomposition(repo_root=repo_root, db_path=db_path, lang="es", limit=1, force=False)
    assert second["status"]["status"] == "skipped"
