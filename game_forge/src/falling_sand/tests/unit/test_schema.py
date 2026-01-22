from falling_sand import schema


def test_migrate_document_dict_v1() -> None:
    payload = {
        "generated_at": "2024-01-01T00:00:00Z",
        "source_root": "src",
        "tests_root": "tests",
        "stats": {"total": 0},
        "entries": [],
    }

    migrated = schema.migrate_document_dict(payload)

    assert migrated["schema_version"] == schema.CURRENT_SCHEMA_VERSION
    assert "test_summary" in migrated
    assert "profile_summary" in migrated
    assert "benchmark_summary" in migrated


def test_migrate_document_dict_current() -> None:
    payload = {
        "schema_version": schema.CURRENT_SCHEMA_VERSION,
        "generated_at": "2024-01-01T00:00:00Z",
        "source_root": "src",
        "tests_root": "tests",
        "stats": {"total": 0},
        "entries": [],
        "test_summary": None,
        "profile_summary": None,
        "benchmark_summary": None,
    }

    migrated = schema.migrate_document_dict(payload)
    assert migrated is payload


def test_migrate_document_dict_v2_to_v3() -> None:
    payload = {
        "schema_version": 2,
        "generated_at": "2024-01-01T00:00:00Z",
        "source_root": "src",
        "tests_root": "tests",
        "stats": {"total": 0},
        "entries": [],
        "benchmark_summary": {
            "runs": 1,
            "mean_seconds": 0.1,
            "median_seconds": 0.1,
            "stdev_seconds": 0.0,
            "min_seconds": 0.1,
            "max_seconds": 0.1,
        },
    }

    migrated = schema.migrate_document_dict(payload)

    assert migrated["schema_version"] == 3
    assert migrated["benchmark_summary"]["benchmarks"][0]["runs"] == 1
