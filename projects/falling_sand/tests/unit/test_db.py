import json
from pathlib import Path

import pytest

from falling_sand import db, models


def _sample_document() -> models.IndexDocument:
    entry = models.IndexEntry(
        name="demo",
        qualname="demo",
        kind="function",
        origin="source",
        module="pkg",
        filepath="pkg.py",
        lineno=1,
        docstring=None,
        signature="()",
    )
    test_case = models.TestCaseResult(
        name="test_demo",
        classname="suite",
        file="tests/test_demo.py",
        line=10,
        duration_seconds=0.1,
        outcome="passed",
    )
    test_summary = models.TestSummary(
        total=1,
        passed=1,
        failed=0,
        skipped=0,
        errors=0,
        duration_seconds=0.1,
        cases=(test_case,),
    )
    profile_stat = models.ProfileFunctionStat(
        function="demo",
        filename="pkg.py",
        lineno=1,
        call_count=1,
        total_time=0.01,
        cumulative_time=0.01,
    )
    profile_summary = models.ProfileSummary(
        total_calls=1,
        total_time=0.01,
        top_functions=(profile_stat,),
    )
    benchmark_case = models.BenchmarkCase(
        name="indexer",
        runs=1,
        mean_seconds=0.02,
        median_seconds=0.02,
        stdev_seconds=0.0,
        min_seconds=0.02,
        max_seconds=0.02,
    )
    benchmark_summary = models.BenchmarkSummary(cases=(benchmark_case,))

    return models.IndexDocument(
        schema_version=2,
        generated_at="2024-01-01T00:00:00Z",
        source_root="src",
        tests_root="tests",
        stats={"total": 1},
        entries=(entry,),
        test_summary=test_summary,
        profile_summary=profile_summary,
        benchmark_summary=benchmark_summary,
    )


def test_migrate_db_creates_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    connection = db.connect_db(db.DbConfig(path=db_path))
    try:
        db.migrate_db(connection)
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()

    assert "runs" in tables
    assert "entries" in tables
    assert "test_summary" in tables
    assert "profile_summary" in tables
    assert "benchmark_summary" in tables
    assert "benchmark_cases" in tables


def test_migrate_db_creates_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    connection = db.connect_db(db.DbConfig(path=db_path))
    try:
        db.migrate_db(connection)
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        )
        indexes = {row[0] for row in cursor.fetchall()}
    finally:
        connection.close()

    assert "idx_entries_run_id" in indexes
    assert "idx_profile_functions_function" in indexes


def test_ingest_document_inserts_rows(tmp_path: Path) -> None:
    document = _sample_document()
    db_path = tmp_path / "index.db"
    connection = db.connect_db(db.DbConfig(path=db_path))
    try:
        db.migrate_db(connection)
        run_id = db.ingest_document(connection, document)
        cursor = connection.execute("SELECT count(*) FROM entries WHERE run_id = ?", (run_id,))
        entries_count = cursor.fetchone()[0]
        cursor = connection.execute("SELECT count(*) FROM test_cases WHERE run_id = ?", (run_id,))
        cases_count = cursor.fetchone()[0]
    finally:
        connection.close()

    assert run_id > 0
    assert entries_count == 1
    assert cases_count == 1


def test_insert_run_stats_json(tmp_path: Path) -> None:
    document = _sample_document()
    db_path = tmp_path / "index.db"
    connection = db.connect_db(db.DbConfig(path=db_path))
    try:
        db.migrate_db(connection)
        run_id = db.insert_run(connection, document)
        cursor = connection.execute("SELECT stats_json FROM runs WHERE id = ?", (run_id,))
        stats_json = cursor.fetchone()[0]
    finally:
        connection.close()

    assert json.loads(stats_json) == {"total": 1}


def test_connect_db_requires_path() -> None:
    with pytest.raises(ValueError, match="path must be a file path"):
        db.DbConfig(path=Path("."))


def test_db_config_validation() -> None:
    with pytest.raises(ValueError, match="journal_mode must be a valid SQLite mode"):
        db.DbConfig(path=Path("index.db"), journal_mode="BAD")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="synchronous must be a valid SQLite mode"):
        db.DbConfig(path=Path("index.db"), synchronous="FAST")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cache_size_kb must be positive"):
        db.DbConfig(path=Path("index.db"), cache_size_kb=0)

    with pytest.raises(ValueError, match="temp_store must be a valid SQLite setting"):
        db.DbConfig(path=Path("index.db"), temp_store="BAD")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="busy_timeout_ms must be non-negative"):
        db.DbConfig(path=Path("index.db"), busy_timeout_ms=-1)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        db.DbConfig(path=Path("index.db"), batch_size=0)


def test_insert_entries_batches(tmp_path: Path) -> None:
    document = _sample_document()
    db_path = tmp_path / "index.db"
    connection = db.connect_db(db.DbConfig(path=db_path, batch_size=1))
    try:
        db.migrate_db(connection)
        run_id = db.insert_run(connection, document)
        entries = list(document.entries) * 3
        db.insert_entries(connection, run_id, entries, batch_size=1)
        cursor = connection.execute("SELECT count(*) FROM entries WHERE run_id = ?", (run_id,))
        count = cursor.fetchone()[0]
    finally:
        connection.close()

    assert count == 3
