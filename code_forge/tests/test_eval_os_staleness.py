from __future__ import annotations

from pathlib import Path

from code_forge.eval_os.staleness import (
    FreshnessRecord,
    compute_staleness_metrics,
    load_freshness_records,
)


def test_compute_staleness_metrics_basic() -> None:
    rows = [
        FreshnessRecord(
            memory_key="a",
            source_last_modified="2026-02-21T10:00:00Z",
            derived_at="2026-02-21T10:05:00Z",
            served_at="2026-02-21T10:06:00Z",
            revalidated_at="2026-02-21T10:06:30Z",
        ),
        FreshnessRecord(
            memory_key="b",
            source_last_modified="2026-02-21T10:10:00Z",
            derived_at="2026-02-21T10:01:00Z",
            served_at="2026-02-21T10:11:00Z",
            revalidated_at="2026-02-21T10:12:00Z",
            stale_error=True,
        ),
    ]
    metrics = compute_staleness_metrics(rows)
    assert metrics["record_count"] == 2
    assert metrics["stale_serve_count"] == 1
    assert metrics["stale_error_count"] == 1
    assert metrics["stale_serve_rate"] > 0.0
    assert metrics["staleness_caused_error_rate"] == 1.0


def test_load_freshness_records_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "freshness.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"memory_key":"a","source_last_modified":"2026-02-21T10:00:00Z","derived_at":"2026-02-21T10:00:01Z","served_at":"2026-02-21T10:00:02Z"}',
                '{"memory_key":"b","source_last_modified":"2026-02-21T10:00:00Z","derived_at":"2026-02-21T09:59:00Z","served_at":"2026-02-21T10:00:03Z"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_freshness_records(path)
    assert len(rows) == 2
    assert rows[0].memory_key == "a"
