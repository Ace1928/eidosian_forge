"""Reporting utilities for performance and testing trends."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from falling_sand.db import DbConfig, connect_db, migrate_db
from eidosian_core import eidosian


@dataclass(frozen=True)
class RunInfo:
    """Metadata for an indexed run."""

    run_id: int
    generated_at: str
    schema_version: int


@dataclass(frozen=True)
class TrendPoint:
    """Single point in a time series."""

    run_id: int
    generated_at: str
    value: float


@dataclass(frozen=True)
class TestTrendPoint:
    """Test summary trend point."""

    run_id: int
    generated_at: str
    total: int
    failed: int
    errors: int
    skipped: int
    duration_seconds: float


@dataclass(frozen=True)
class Hotspot:
    """Aggregated performance hotspot across runs."""

    function: str
    filename: str
    avg_cumulative_time: float
    avg_total_time: float
    avg_call_count: float
    samples: int


@dataclass(frozen=True)
class BenchmarkTrend:
    """Benchmark trend series for a named benchmark."""

    name: str
    points: tuple[TrendPoint, ...]

    @eidosian()
    def to_dict(self) -> dict[str, object]:
        """Serialize the benchmark trend for JSON output."""

        return {"name": self.name, "points": [point.__dict__ for point in self.points]}


@dataclass(frozen=True)
class ReportDocument:
    """Full reporting document for database trends."""

    schema_version: int
    generated_at: str
    run_count: int
    runs: tuple[RunInfo, ...]
    benchmark_trends: tuple[BenchmarkTrend, ...]
    test_trend: tuple[TestTrendPoint, ...]
    hotspots: tuple[Hotspot, ...]

    @eidosian()
    def to_dict(self) -> dict[str, object]:
        """Serialize the report for JSON output."""

        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "run_count": self.run_count,
            "runs": [run.__dict__ for run in self.runs],
            "benchmark_trends": [trend.to_dict() for trend in self.benchmark_trends],
            "test_trend": [point.__dict__ for point in self.test_trend],
            "hotspots": [hotspot.__dict__ for hotspot in self.hotspots],
        }


def _fetch_runs(connection: sqlite3.Connection, run_limit: int) -> list[RunInfo]:
    cursor = connection.execute(
        """
        SELECT id, generated_at, schema_version
        FROM runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (run_limit,),
    )
    rows = cursor.fetchall()
    runs = [RunInfo(run_id=row[0], generated_at=row[1], schema_version=row[2]) for row in rows]
    return list(reversed(runs))


def _fetch_benchmark_trends(
    connection: sqlite3.Connection, run_ids: Sequence[int]
) -> list[BenchmarkTrend]:
    if not run_ids:
        return []
    placeholders = ",".join("?" for _ in run_ids)
    cursor = connection.execute(
        f"""
        SELECT benchmark_cases.name, runs.id, runs.generated_at, benchmark_cases.mean_seconds
        FROM benchmark_cases
        JOIN runs ON runs.id = benchmark_cases.run_id
        WHERE runs.id IN ({placeholders})
        ORDER BY benchmark_cases.name ASC, runs.id ASC
        """,
        run_ids,
    )
    rows = cursor.fetchall()
    trends: dict[str, list[TrendPoint]] = {}
    for name, run_id, generated_at, mean_seconds in rows:
        trends.setdefault(name, []).append(
            TrendPoint(run_id=run_id, generated_at=generated_at, value=mean_seconds)
        )
    return [BenchmarkTrend(name=name, points=tuple(points)) for name, points in trends.items()]


def _fetch_test_trend(connection: sqlite3.Connection, run_ids: Sequence[int]) -> list[TestTrendPoint]:
    if not run_ids:
        return []
    placeholders = ",".join("?" for _ in run_ids)
    cursor = connection.execute(
        f"""
        SELECT runs.id, runs.generated_at, test_summary.total, test_summary.failed,
               test_summary.errors, test_summary.skipped, test_summary.duration_seconds
        FROM test_summary
        JOIN runs ON runs.id = test_summary.run_id
        WHERE runs.id IN ({placeholders})
        ORDER BY runs.id ASC
        """,
        run_ids,
    )
    return [
        TestTrendPoint(
            run_id=row[0],
            generated_at=row[1],
            total=row[2],
            failed=row[3],
            errors=row[4],
            skipped=row[5],
            duration_seconds=row[6],
        )
        for row in cursor.fetchall()
    ]


def _fetch_hotspots(
    connection: sqlite3.Connection, run_ids: Sequence[int], top_n: int
) -> list[Hotspot]:
    if not run_ids:
        return []
    placeholders = ",".join("?" for _ in run_ids)
    cursor = connection.execute(
        f"""
        SELECT function, filename,
               AVG(cumulative_time) AS avg_cumulative_time,
               AVG(total_time) AS avg_total_time,
               AVG(call_count) AS avg_call_count,
               COUNT(*) AS samples
        FROM profile_functions
        WHERE run_id IN ({placeholders})
        GROUP BY function, filename
        ORDER BY avg_cumulative_time DESC
        LIMIT ?
        """,
        (*run_ids, top_n),
    )
    return [
        Hotspot(
            function=row[0],
            filename=row[1],
            avg_cumulative_time=row[2],
            avg_total_time=row[3],
            avg_call_count=row[4],
            samples=row[5],
        )
        for row in cursor.fetchall()
    ]


@eidosian()
def generate_report(db_path: Path, run_limit: int, top_n: int) -> ReportDocument:
    """Generate a report document from the SQLite database."""

    if run_limit <= 0:
        raise ValueError("run_limit must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not db_path.exists():
        raise ValueError(f"Database not found: {db_path}")

    connection = connect_db(DbConfig(path=db_path))
    try:
        migrate_db(connection)
        runs = _fetch_runs(connection, run_limit)
        run_ids = [run.run_id for run in runs]

        benchmark_trends = _fetch_benchmark_trends(connection, run_ids)
        test_trend = _fetch_test_trend(connection, run_ids)
        hotspots = _fetch_hotspots(connection, run_ids, top_n)
    finally:
        connection.close()

    return ReportDocument(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        run_count=len(runs),
        runs=tuple(runs),
        benchmark_trends=tuple(benchmark_trends),
        test_trend=tuple(test_trend),
        hotspots=tuple(hotspots),
    )


@eidosian()
def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for reporting."""

    parser = argparse.ArgumentParser(description="Generate trend reports from SQLite.")
    parser.add_argument("--db", type=Path, default=Path("artifacts/index.db"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/report.json"))
    parser.add_argument("--run-limit", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=10)
    return parser


@eidosian()
def main(argv: Sequence[str] | None = None) -> int:
    """Run the reporting CLI."""

    args = build_parser().parse_args(argv)
    report = generate_report(args.db, args.run_limit, args.top_n)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
