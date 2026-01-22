import pytest

from falling_sand import models


def test_index_entry_to_dict() -> None:
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

    payload = entry.to_dict()
    assert payload["name"] == "demo"
    assert payload["kind"] == "function"


def test_test_summary_to_dict() -> None:
    case = models.TestCaseResult(
        name="test_ok",
        classname="suite",
        file=None,
        line=None,
        duration_seconds=0.1,
        outcome="passed",
    )
    summary = models.TestSummary(
        total=1,
        passed=1,
        failed=0,
        skipped=0,
        errors=0,
        duration_seconds=0.1,
        cases=(case,),
    )

    payload = summary.to_dict()
    assert payload["total"] == 1
    assert payload["cases"][0]["outcome"] == "passed"


def test_profile_summary_to_dict() -> None:
    stat = models.ProfileFunctionStat(
        function="demo",
        filename="demo.py",
        lineno=10,
        call_count=1,
        total_time=0.1,
        cumulative_time=0.1,
    )
    summary = models.ProfileSummary(total_calls=1, total_time=0.1, top_functions=(stat,))

    payload = summary.to_dict()
    assert payload["total_calls"] == 1
    assert payload["top_functions"][0]["function"] == "demo"


def test_benchmark_summary_to_dict() -> None:
    case = models.BenchmarkCase(
        name="indexer",
        runs=2,
        mean_seconds=0.2,
        median_seconds=0.2,
        stdev_seconds=0.0,
        min_seconds=0.1,
        max_seconds=0.3,
    )
    summary = models.BenchmarkSummary(cases=(case,))

    payload = summary.to_dict()
    assert payload["benchmarks"][0]["runs"] == 2
    assert payload["benchmarks"][0]["min_seconds"] == 0.1


def test_index_document_to_dict() -> None:
    entry = models.IndexEntry(
        name="demo",
        qualname="demo",
        kind="function",
        origin="source",
        module="pkg",
        filepath="pkg.py",
        lineno=1,
        docstring=None,
        signature=None,
    )
    summary = models.TestSummary(
        total=1,
        passed=1,
        failed=0,
        skipped=0,
        errors=0,
        duration_seconds=0.1,
        cases=(),
    )
    document = models.IndexDocument(
        schema_version=2,
        generated_at="2024-01-01T00:00:00Z",
        source_root="src",
        tests_root="tests",
        stats={"total": 1},
        entries=(entry,),
        test_summary=summary,
        profile_summary=None,
        benchmark_summary=None,
    )

    payload = document.to_dict()
    assert payload["schema_version"] == 2
    assert payload["entries"][0]["name"] == "demo"


def test_index_document_round_trip() -> None:
    entry = models.IndexEntry(
        name="demo",
        qualname="demo",
        kind="function",
        origin="source",
        module="pkg",
        filepath="pkg.py",
        lineno=1,
        docstring=None,
        signature=None,
    )
    document = models.IndexDocument(
        schema_version=2,
        generated_at="2024-01-01T00:00:00Z",
        source_root="src",
        tests_root="tests",
        stats={"total": 1},
        entries=(entry,),
        test_summary=None,
        profile_summary=None,
        benchmark_summary=None,
    )

    round_trip = models.index_document_from_dict(document.to_dict())
    assert round_trip.schema_version == document.schema_version
    assert round_trip.entries[0].name == "demo"


def test_index_document_stats_validation() -> None:
    payload = {
        "schema_version": 2,
        "generated_at": "2024-01-01T00:00:00Z",
        "source_root": "src",
        "tests_root": "tests",
        "stats": {"total": "1"},
        "entries": [],
        "test_summary": None,
        "profile_summary": None,
        "benchmark_summary": None,
    }

    document = models.index_document_from_dict(payload)
    assert document.stats == {"total": 1}


def test_from_dict_helpers() -> None:
    entry_payload = {
        "name": "demo",
        "qualname": "demo",
        "kind": "function",
        "origin": "source",
        "module": "pkg",
        "filepath": "pkg.py",
        "lineno": 1,
        "docstring": None,
        "signature": "()",
    }
    entry = models.index_entry_from_dict(entry_payload)
    assert entry.name == "demo"

    test_summary_payload = {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration_seconds": 0.1,
        "cases": [
            {
                "name": "test_demo",
                "classname": "suite",
                "file": "tests/test_demo.py",
                "line": 10,
                "duration_seconds": 0.1,
                "outcome": "passed",
            }
        ],
    }
    test_summary = models.test_summary_from_dict(test_summary_payload)
    assert test_summary.total == 1
    assert test_summary.cases[0].name == "test_demo"

    profile_summary_payload = {
        "total_calls": 1,
        "total_time": 0.2,
        "top_functions": [
            {
                "function": "demo",
                "filename": "pkg.py",
                "lineno": 1,
                "call_count": 1,
                "total_time": 0.1,
                "cumulative_time": 0.2,
            }
        ],
    }
    profile_summary = models.profile_summary_from_dict(profile_summary_payload)
    assert profile_summary.total_calls == 1
    assert profile_summary.top_functions[0].function == "demo"

    benchmark_summary_payload = {
        "benchmarks": [
            {
                "name": "simulation",
                "runs": 2,
                "mean_seconds": 0.1,
                "median_seconds": 0.1,
                "stdev_seconds": 0.0,
                "min_seconds": 0.1,
                "max_seconds": 0.2,
            }
        ]
    }
    benchmark_summary = models.benchmark_summary_from_dict(benchmark_summary_payload)
    assert benchmark_summary.cases[0].runs == 2


def test_benchmark_summary_legacy_payload() -> None:
    payload = {
        "runs": 1,
        "mean_seconds": 0.1,
        "median_seconds": 0.1,
        "stdev_seconds": 0.0,
        "min_seconds": 0.1,
        "max_seconds": 0.1,
    }
    summary = models.benchmark_summary_from_dict(payload)
    assert summary.cases[0].runs == 1


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        models.IndexEntry(
            name="",
            qualname="demo",
            kind="function",
            origin="source",
            module="pkg",
            filepath="pkg.py",
            lineno=1,
            docstring=None,
            signature=None,
        )


def test_profile_function_lineno_zero_allowed() -> None:
    stat = models.ProfileFunctionStat(
        function="builtin",
        filename="~",
        lineno=0,
        call_count=1,
        total_time=0.0,
        cumulative_time=0.0,
    )
    assert stat.lineno == 0


def test_profile_function_negative_lineno_rejected() -> None:
    with pytest.raises(ValueError, match="lineno must be non-negative"):
        models.ProfileFunctionStat(
            function="bad",
            filename="demo.py",
            lineno=-1,
            call_count=1,
            total_time=0.0,
            cumulative_time=0.0,
        )
