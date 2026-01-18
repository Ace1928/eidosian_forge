import json
from pathlib import Path

import pytest

from falling_sand import db, models, reporting


def _document(run_id: int) -> models.IndexDocument:
    entry = models.IndexEntry(
        name=f"demo_{run_id}",
        qualname=f"demo_{run_id}",
        kind="function",
        origin="source",
        module="pkg",
        filepath="pkg.py",
        lineno=1,
        docstring=None,
        signature="()",
    )
    test_summary = models.TestSummary(
        total=1,
        passed=1,
        failed=0,
        skipped=0,
        errors=0,
        duration_seconds=0.1 * run_id,
        cases=(),
    )
    profile_stat = models.ProfileFunctionStat(
        function="demo",
        filename="pkg.py",
        lineno=1,
        call_count=1,
        total_time=0.01 * run_id,
        cumulative_time=0.02 * run_id,
    )
    profile_summary = models.ProfileSummary(
        total_calls=1,
        total_time=0.02 * run_id,
        top_functions=(profile_stat,),
    )
    benchmark_case = models.BenchmarkCase(
        name="simulation",
        runs=1,
        mean_seconds=0.03 * run_id,
        median_seconds=0.03 * run_id,
        stdev_seconds=0.0,
        min_seconds=0.03 * run_id,
        max_seconds=0.03 * run_id,
    )
    benchmark_summary = models.BenchmarkSummary(cases=(benchmark_case,))

    return models.IndexDocument(
        schema_version=2,
        generated_at=f"2024-01-01T00:00:0{run_id}Z",
        source_root="src",
        tests_root="tests",
        stats={"total": 1},
        entries=(entry,),
        test_summary=test_summary,
        profile_summary=profile_summary,
        benchmark_summary=benchmark_summary,
    )


def test_generate_report(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    connection = db.connect_db(db.DbConfig(path=db_path))
    try:
        db.migrate_db(connection)
        db.ingest_document(connection, _document(1))
        db.ingest_document(connection, _document(2))
    finally:
        connection.close()

    report = reporting.generate_report(db_path, run_limit=10, top_n=5)

    assert report.run_count == 2
    assert report.benchmark_trends
    assert report.hotspots


def test_generate_report_invalid_inputs(tmp_path: Path) -> None:
    db_path = tmp_path / "missing.db"
    with pytest.raises(ValueError, match="Database not found"):
        reporting.generate_report(db_path, run_limit=1, top_n=1)

    db_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="run_limit must be positive"):
        reporting.generate_report(db_path, run_limit=0, top_n=1)


def test_report_cli(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    connection = db.connect_db(db.DbConfig(path=db_path))
    try:
        db.migrate_db(connection)
        db.ingest_document(connection, _document(1))
    finally:
        connection.close()

    output = tmp_path / "report.json"
    exit_code = reporting.main([
        "--db",
        str(db_path),
        "--output",
        str(output),
        "--run-limit",
        "5",
        "--top-n",
        "3",
    ])

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_count"] == 1
    assert payload["benchmark_trends"]
