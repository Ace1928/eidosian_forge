from __future__ import annotations

import json
from pathlib import Path

from word_forge.multilingual.runtime import run_multilingual_ingest


def test_run_multilingual_ingest_writes_status_and_reports(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "data" / "wiktextract_sample.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "word": "bonjour",
                "lang": "fr",
                "pos": "interj",
                "senses": [{"glosses": ["hello"]}],
                "translations": [{"lang": "en", "word": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    result = run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="wiktextract",
        db_path=db_path,
    )

    assert result["status"]["status"] == "completed"
    assert result["report"]["deltas"]["lexeme_delta"] == 1
    assert result["report"]["deltas"]["translation_delta"] == 1
    assert result["report"]["deltas"]["base_aligned_delta"] == 1
    from word_forge.database.database_manager import DBManager
    db = DBManager(db_path=db_path)
    assert db.get_word_id("hello") > 0
    assert (repo_root / "data" / "runtime" / "word_forge_multilingual_ingest_status.json").exists()
    assert (repo_root / "data" / "runtime" / "word_forge_multilingual_ingest_history.jsonl").exists()
    assert Path(result["artifacts"]["latest_json"]).exists()
    assert Path(result["artifacts"]["latest_markdown"]).exists()


def test_run_multilingual_ingest_skips_unchanged_source(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "data" / "kaikki_sample.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "word": "hola",
                "lang": "es",
                "pos": "interj",
                "senses": [{"glosses": ["hello"]}],
                "translations": [{"lang": "en", "word": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    first = run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="kaikki",
        db_path=db_path,
    )
    second = run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="kaikki",
        db_path=db_path,
    )

    assert first["status"]["status"] == "completed"
    assert second["status"]["status"] == "skipped"
    assert second["status"]["phase"] == "unchanged"
