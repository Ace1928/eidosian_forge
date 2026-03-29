from __future__ import annotations

from pathlib import Path

from word_forge.database.database_manager import DBManager
from word_forge.multilingual.fasttext_runtime import run_fasttext_ingest


def _write_vec(path: Path, rows: list[tuple[str, tuple[float, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{len(rows)} 2"]
    for term, (x, y) in rows:
        lines.append(f"{term} {x} {y}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_run_fasttext_ingest_bootstraps_translations(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.insert_or_update_word("archive", "stored material", "noun", [])
    db.insert_or_update_word("atlas", "supporting map", "noun", [])
    db.insert_or_update_lexeme("archivo", "es", source="seed")
    db.insert_or_update_lexeme("atlas", "es", source="seed")

    en_vec = repo_root / "samples" / "en.vec"
    es_vec = repo_root / "samples" / "es.vec"
    _write_vec(en_vec, [("archive", (1.0, 0.0)), ("atlas", (0.95, 0.05))])
    _write_vec(es_vec, [("archivo", (0.99, 0.01)), ("atlas", (0.94, 0.06))])

    first = run_fasttext_ingest(
        repo_root=repo_root,
        source_path=en_vec,
        lang="en",
        db_path=db_path,
        vector_db_path=repo_root / "data" / "word_forge_fasttext.sqlite",
        force=True,
    )
    assert first["status"]["status"] == "completed"

    second = run_fasttext_ingest(
        repo_root=repo_root,
        source_path=es_vec,
        lang="es",
        db_path=db_path,
        vector_db_path=repo_root / "data" / "word_forge_fasttext.sqlite",
        bootstrap_lang="en",
        min_score=0.8,
        apply=True,
        force=True,
    )
    assert second["status"]["status"] == "completed"
    report = second["report"]
    assert report is not None
    assert report["after"]["fasttext"]["vector_count"] == 4
    assert report["after"]["fasttext"]["candidate_count"] >= 2
    assert report["deltas"]["translation_delta"] >= 2
    assert report["deltas"]["base_aligned_delta"] >= 2


def test_run_fasttext_ingest_skips_unchanged_source(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    DBManager(db_path=db_path)
    vec_path = repo_root / "samples" / "en.vec"
    _write_vec(vec_path, [("archive", (1.0, 0.0))])

    result = run_fasttext_ingest(
        repo_root=repo_root,
        source_path=vec_path,
        lang="en",
        db_path=db_path,
        vector_db_path=repo_root / "data" / "word_forge_fasttext.sqlite",
        force=True,
    )
    assert result["status"]["status"] == "completed"

    skipped = run_fasttext_ingest(
        repo_root=repo_root,
        source_path=vec_path,
        lang="en",
        db_path=db_path,
        vector_db_path=repo_root / "data" / "word_forge_fasttext.sqlite",
        force=False,
    )
    assert skipped["status"]["status"] == "skipped"
