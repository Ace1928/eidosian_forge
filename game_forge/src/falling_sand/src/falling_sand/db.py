"""SQLite ingestion for index documents."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

from falling_sand.models import IndexDocument, IndexEntry, TestCaseResult

CURRENT_DB_VERSION = 3

JournalMode = Literal["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"]
SynchronousMode = Literal["OFF", "NORMAL", "FULL", "EXTRA"]


@dataclass(frozen=True)
class DbConfig:
    """Configuration for connecting to the SQLite database."""

    path: Path
    journal_mode: JournalMode = "WAL"
    synchronous: SynchronousMode = "NORMAL"
    cache_size_kb: int | None = 20000
    temp_store: Literal["DEFAULT", "FILE", "MEMORY"] = "MEMORY"
    busy_timeout_ms: int = 5000
    batch_size: int = 1000

    def __post_init__(self) -> None:
        if not self.path or self.path.name in {"", "."}:
            raise ValueError("path must be a file path")
        if self.journal_mode not in {"DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"}:
            raise ValueError("journal_mode must be a valid SQLite mode")
        if self.synchronous not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError("synchronous must be a valid SQLite mode")
        if self.cache_size_kb is not None and self.cache_size_kb <= 0:
            raise ValueError("cache_size_kb must be positive when provided")
        if self.temp_store not in {"DEFAULT", "FILE", "MEMORY"}:
            raise ValueError("temp_store must be a valid SQLite setting")
        if self.busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def connect_db(config: DbConfig) -> sqlite3.Connection:
    """Connect to the SQLite database with foreign keys enabled."""

    connection = sqlite3.connect(str(config.path))
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA journal_mode = {config.journal_mode}")
    connection.execute(f"PRAGMA synchronous = {config.synchronous}")
    if config.cache_size_kb is not None:
        connection.execute(f"PRAGMA cache_size = {-config.cache_size_kb}")
    connection.execute(f"PRAGMA temp_store = {config.temp_store}")
    connection.execute(f"PRAGMA busy_timeout = {config.busy_timeout_ms}")
    return connection


def _batched(rows: Iterable[tuple[object, ...]], batch_size: int) -> Iterator[list[tuple[object, ...]]]:
    batch: list[tuple[object, ...]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _ensure_migrations_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )


def _applied_versions(connection: sqlite3.Connection) -> set[int]:
    cursor = connection.execute("SELECT version FROM schema_migrations")
    return {row[0] for row in cursor.fetchall()}


def _migrations() -> list[tuple[int, str]]:
    return [
        (
            1,
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                generated_at TEXT NOT NULL,
                source_root TEXT NOT NULL,
                tests_root TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                stats_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                qualname TEXT NOT NULL,
                kind TEXT NOT NULL,
                origin TEXT NOT NULL,
                module TEXT NOT NULL,
                filepath TEXT NOT NULL,
                lineno INTEGER NOT NULL,
                docstring TEXT,
                signature TEXT
            );

            CREATE TABLE IF NOT EXISTS test_summary (
                run_id INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                total INTEGER NOT NULL,
                passed INTEGER NOT NULL,
                failed INTEGER NOT NULL,
                skipped INTEGER NOT NULL,
                errors INTEGER NOT NULL,
                duration_seconds REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS test_cases (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                classname TEXT,
                file TEXT,
                line INTEGER,
                duration_seconds REAL NOT NULL,
                outcome TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS profile_summary (
                run_id INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                total_calls INTEGER NOT NULL,
                total_time REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS profile_functions (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                function TEXT NOT NULL,
                filename TEXT NOT NULL,
                lineno INTEGER NOT NULL,
                call_count INTEGER NOT NULL,
                total_time REAL NOT NULL,
                cumulative_time REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS benchmark_summary (
                run_id INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                runs INTEGER NOT NULL,
                mean_seconds REAL NOT NULL,
                median_seconds REAL NOT NULL,
                stdev_seconds REAL NOT NULL,
                min_seconds REAL NOT NULL,
                max_seconds REAL NOT NULL
            );
            """,
        ),
        (
            2,
            """
            CREATE INDEX IF NOT EXISTS idx_entries_run_id ON entries(run_id);
            CREATE INDEX IF NOT EXISTS idx_test_cases_run_id ON test_cases(run_id);
            CREATE INDEX IF NOT EXISTS idx_profile_functions_run_id ON profile_functions(run_id);
            CREATE INDEX IF NOT EXISTS idx_profile_functions_function ON profile_functions(function);
            CREATE INDEX IF NOT EXISTS idx_benchmark_summary_run_id ON benchmark_summary(run_id);
            """,
        ),
        (
            3,
            """
            CREATE TABLE IF NOT EXISTS benchmark_cases (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                runs INTEGER NOT NULL,
                mean_seconds REAL NOT NULL,
                median_seconds REAL NOT NULL,
                stdev_seconds REAL NOT NULL,
                min_seconds REAL NOT NULL,
                max_seconds REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_benchmark_cases_run_id ON benchmark_cases(run_id);
            CREATE INDEX IF NOT EXISTS idx_benchmark_cases_name ON benchmark_cases(name);
            """,
        ),
    ]


def migrate_db(connection: sqlite3.Connection) -> None:
    """Apply database migrations to the latest version."""

    _ensure_migrations_table(connection)
    applied = _applied_versions(connection)

    for version, ddl in _migrations():
        if version in applied:
            continue
        with connection:
            connection.executescript(ddl)
            connection.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, datetime('now'))",
                (version,),
            )


def insert_run(connection: sqlite3.Connection, document: IndexDocument) -> int:
    """Insert a run and return its ID."""

    cursor = connection.execute(
        """
        INSERT INTO runs (generated_at, source_root, tests_root, schema_version, stats_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            document.generated_at,
            document.source_root,
            document.tests_root,
            document.schema_version,
            json.dumps(document.stats, sort_keys=True),
        ),
    )
    if cursor.lastrowid is None:
        raise ValueError("Failed to insert run")
    return int(cursor.lastrowid)


def insert_entries(
    connection: sqlite3.Connection,
    run_id: int,
    entries: Iterable[IndexEntry],
    batch_size: int = 1000,
) -> None:
    """Insert index entries for a run."""

    rows = [
        (
            run_id,
            entry.name,
            entry.qualname,
            entry.kind,
            entry.origin,
            entry.module,
            entry.filepath,
            entry.lineno,
            entry.docstring,
            entry.signature,
        )
        for entry in entries
    ]
    if not rows:
        return
    for batch in _batched(rows, batch_size):
        connection.executemany(
            """
            INSERT INTO entries (
                run_id, name, qualname, kind, origin, module, filepath, lineno, docstring, signature
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )


def insert_test_summary(
    connection: sqlite3.Connection,
    run_id: int,
    total: int,
    passed: int,
    failed: int,
    skipped: int,
    errors: int,
    duration_seconds: float,
    cases: Iterable[TestCaseResult],
    batch_size: int = 1000,
) -> None:
    """Insert test summary and cases for a run."""

    connection.execute(
        """
        INSERT INTO test_summary (run_id, total, passed, failed, skipped, errors, duration_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, total, passed, failed, skipped, errors, duration_seconds),
    )

    rows = [
        (
            run_id,
            case.name,
            case.classname,
            case.file,
            case.line,
            case.duration_seconds,
            case.outcome,
        )
        for case in cases
    ]
    if not rows:
        return
    for batch in _batched(rows, batch_size):
        connection.executemany(
            """
            INSERT INTO test_cases (
                run_id, name, classname, file, line, duration_seconds, outcome
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )


def insert_profile_summary(
    connection: sqlite3.Connection,
    run_id: int,
    total_calls: int,
    total_time: float,
    functions: Iterable[tuple[str, str, int, int, float, float]],
    batch_size: int = 1000,
) -> None:
    """Insert profiling summary and function stats."""

    connection.execute(
        """
        INSERT INTO profile_summary (run_id, total_calls, total_time)
        VALUES (?, ?, ?)
        """,
        (run_id, total_calls, total_time),
    )

    rows = [(run_id, *function) for function in functions]
    if not rows:
        return
    for batch in _batched(rows, batch_size):
        connection.executemany(
            """
            INSERT INTO profile_functions (
                run_id, function, filename, lineno, call_count, total_time, cumulative_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )


def insert_benchmark_summary(
    connection: sqlite3.Connection,
    run_id: int,
    cases: Iterable[tuple[str, int, float, float, float, float, float]],
    batch_size: int = 1000,
) -> None:
    """Insert benchmark summary for a run."""

    rows = [(run_id, *case) for case in cases]
    if not rows:
        return
    for batch in _batched(rows, batch_size):
        connection.executemany(
            """
            INSERT INTO benchmark_cases (
                run_id, name, runs, mean_seconds, median_seconds, stdev_seconds, min_seconds, max_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )


def ingest_document(connection: sqlite3.Connection, document: IndexDocument, batch_size: int = 1000) -> int:
    """Insert a full index document and return the run ID."""

    with connection:
        run_id = insert_run(connection, document)
        insert_entries(connection, run_id, document.entries, batch_size=batch_size)

        if document.test_summary:
            summary = document.test_summary
            insert_test_summary(
                connection,
                run_id,
                summary.total,
                summary.passed,
                summary.failed,
                summary.skipped,
                summary.errors,
                summary.duration_seconds,
                summary.cases,
                batch_size=batch_size,
            )

        if document.profile_summary:
            profile_summary = document.profile_summary
            functions = [
                (
                    stat.function,
                    stat.filename,
                    stat.lineno,
                    stat.call_count,
                    stat.total_time,
                    stat.cumulative_time,
                )
                for stat in profile_summary.top_functions
            ]
            insert_profile_summary(
                connection,
                run_id,
                profile_summary.total_calls,
                profile_summary.total_time,
                functions,
                batch_size=batch_size,
            )

        if document.benchmark_summary:
            benchmark_summary = document.benchmark_summary
            cases = [
                (
                    case.name,
                    case.runs,
                    case.mean_seconds,
                    case.median_seconds,
                    case.stdev_seconds,
                    case.min_seconds,
                    case.max_seconds,
                )
                for case in benchmark_summary.cases
            ]
            insert_benchmark_summary(
                connection,
                run_id,
                cases,
                batch_size=batch_size,
            )

    return run_id
