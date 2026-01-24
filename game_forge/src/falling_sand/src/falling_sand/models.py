"""Data models for indexing and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from eidosian_core import eidosian

Origin = Literal["source", "test"]
Kind = Literal["function", "class", "method"]
TestOutcome = Literal["passed", "failed", "skipped", "error"]


def _require_non_empty(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _require_non_negative(name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _as_str(value: object, field: str, *, allow_none: bool = False) -> str | None:
    if value is None:
        return None if allow_none else ""
    if isinstance(value, str):
        return value
    raise ValueError(f"{field} must be a string")


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"{field} must be an integer")


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a float")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"{field} must be a float")


def _as_dict(value: object, field: str) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"{field} must be a JSON object")


def _as_list(value: object, field: str) -> list[object]:
    if isinstance(value, list):
        return value
    raise ValueError(f"{field} must be a list")


def _as_int_dict(value: object, field: str) -> dict[str, int]:
    payload = _as_dict(value, field)
    result: dict[str, int] = {}
    for key, item in payload.items():
        if not isinstance(key, str):
            raise ValueError(f"{field} keys must be strings")
        result[key] = _as_int(item, f"{field}.{key}")
    return result


@dataclass(frozen=True)
class IndexEntry:
    """Represents a single code symbol discovered in the project."""

    name: str
    qualname: str
    kind: Kind
    origin: Origin
    module: str
    filepath: str
    lineno: int
    docstring: str | None
    signature: str | None

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        _require_non_empty("qualname", self.qualname)
        _require_non_empty("module", self.module)
        _require_non_empty("filepath", self.filepath)
        if self.lineno <= 0:
            raise ValueError("lineno must be positive")

    @eidosian()
    def to_dict(self) -> dict[str, object]:
        """Serialize the entry for JSON output."""

        return {
            "name": self.name,
            "qualname": self.qualname,
            "kind": self.kind,
            "origin": self.origin,
            "module": self.module,
            "filepath": self.filepath,
            "lineno": self.lineno,
            "docstring": self.docstring,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class TestCaseResult:
    """Represents a single test case execution result."""

    name: str
    classname: str | None
    file: str | None
    line: int | None
    duration_seconds: float
    outcome: TestOutcome

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        _require_non_negative("duration_seconds", self.duration_seconds)

    @eidosian()
    def to_dict(self) -> dict[str, object]:
        """Serialize the test case result for JSON output."""

        return {
            "name": self.name,
            "classname": self.classname,
            "file": self.file,
            "line": self.line,
            "duration_seconds": self.duration_seconds,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class TestSummary:
    """Summary of test results."""

    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_seconds: float
    cases: tuple[TestCaseResult, ...]

    def __post_init__(self) -> None:
        _require_non_negative("total", self.total)
        _require_non_negative("passed", self.passed)
        _require_non_negative("failed", self.failed)
        _require_non_negative("skipped", self.skipped)
        _require_non_negative("errors", self.errors)
        _require_non_negative("duration_seconds", self.duration_seconds)
        if self.total != self.passed + self.failed + self.skipped + self.errors:
            raise ValueError("test summary counts do not add up")

    @eidosian()
    def to_dict(self) -> dict[str, object]:
        """Serialize the test summary for JSON output."""

        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class ProfileFunctionStat:
    """Performance statistics for a single function."""

    function: str
    filename: str
    lineno: int
    call_count: int
    total_time: float
    cumulative_time: float

    def __post_init__(self) -> None:
        _require_non_empty("function", self.function)
        _require_non_empty("filename", self.filename)
        _require_non_negative("call_count", self.call_count)
        _require_non_negative("total_time", self.total_time)
        _require_non_negative("cumulative_time", self.cumulative_time)
        if self.lineno < 0:
            raise ValueError("lineno must be non-negative")

    @eidosian()
    def to_dict(self) -> dict[str, object]:
        """Serialize the function stats for JSON output."""

        return {
            "function": self.function,
            "filename": self.filename,
            "lineno": self.lineno,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "cumulative_time": self.cumulative_time,
        }


@dataclass(frozen=True)
class ProfileSummary:
    """Summary of profiling statistics."""

    total_calls: int
    total_time: float
    top_functions: tuple[ProfileFunctionStat, ...]

    def __post_init__(self) -> None:
        _require_non_negative("total_calls", self.total_calls)
        _require_non_negative("total_time", self.total_time)

    @eidosian()
    def to_dict(self) -> dict[str, object]:
        """Serialize the profile summary for JSON output."""

        return {
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "top_functions": [stat.to_dict() for stat in self.top_functions],
        }


@dataclass(frozen=True)
class BenchmarkCase:
    """Statistics for a single named benchmark."""

    name: str
    runs: int
    mean_seconds: float
    median_seconds: float
    stdev_seconds: float
    min_seconds: float
    max_seconds: float

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        _require_non_negative("runs", self.runs)
        _require_non_negative("mean_seconds", self.mean_seconds)
        _require_non_negative("median_seconds", self.median_seconds)
        _require_non_negative("stdev_seconds", self.stdev_seconds)
        _require_non_negative("min_seconds", self.min_seconds)
        _require_non_negative("max_seconds", self.max_seconds)

    @eidosian()
    def to_dict(self) -> dict[str, object]:
        """Serialize the benchmark case for JSON output."""

        return {
            "name": self.name,
            "runs": self.runs,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "stdev_seconds": self.stdev_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
        }


@dataclass(frozen=True)
class BenchmarkSummary:
    """Summary of benchmark statistics."""

    cases: tuple[BenchmarkCase, ...]

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("cases must be non-empty")

    @eidosian()
    def to_dict(self) -> dict[str, object]:
        """Serialize the benchmark summary for JSON output."""

        return {
            "benchmarks": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class IndexDocument:
    """Full index document including schema version and optional reports."""

    schema_version: int
    generated_at: str
    source_root: str
    tests_root: str
    stats: dict[str, int]
    entries: tuple[IndexEntry, ...]
    test_summary: TestSummary | None
    profile_summary: ProfileSummary | None
    benchmark_summary: BenchmarkSummary | None

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        _require_non_empty("generated_at", self.generated_at)
        _require_non_empty("source_root", self.source_root)
        _require_non_empty("tests_root", self.tests_root)
        if not self.stats:
            raise ValueError("stats must be non-empty")

    @eidosian()
    def to_dict(self) -> dict[str, object]:
        """Serialize the document for JSON output."""

        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "source_root": self.source_root,
            "tests_root": self.tests_root,
            "stats": self.stats,
            "entries": [entry.to_dict() for entry in self.entries],
            "test_summary": self.test_summary.to_dict() if self.test_summary else None,
            "profile_summary": self.profile_summary.to_dict() if self.profile_summary else None,
            "benchmark_summary": self.benchmark_summary.to_dict() if self.benchmark_summary else None,
        }


@eidosian()
def index_entry_from_dict(payload: dict[str, object]) -> IndexEntry:
    """Deserialize an index entry from JSON payload."""

    payload = _as_dict(payload, "entry")
    return IndexEntry(
        name=_as_str(payload.get("name"), "name") or "",
        qualname=_as_str(payload.get("qualname"), "qualname") or "",
        kind=_as_str(payload.get("kind"), "kind") or "function",  # type: ignore[arg-type]
        origin=_as_str(payload.get("origin"), "origin") or "source",  # type: ignore[arg-type]
        module=_as_str(payload.get("module"), "module") or "",
        filepath=_as_str(payload.get("filepath"), "filepath") or "",
        lineno=_as_int(payload.get("lineno", 0), "lineno"),
        docstring=_as_str(payload.get("docstring"), "docstring", allow_none=True),
        signature=_as_str(payload.get("signature"), "signature", allow_none=True),
    )


@eidosian()
def test_case_from_dict(payload: dict[str, object]) -> TestCaseResult:
    """Deserialize a test case result from JSON payload."""

    payload = _as_dict(payload, "test_case")
    return TestCaseResult(
        name=_as_str(payload.get("name"), "name") or "",
        classname=_as_str(payload.get("classname"), "classname", allow_none=True),
        file=_as_str(payload.get("file"), "file", allow_none=True),
        line=_as_int(payload.get("line"), "line") if payload.get("line") is not None else None,
        duration_seconds=_as_float(payload.get("duration_seconds", 0.0), "duration_seconds"),
        outcome=_as_str(payload.get("outcome"), "outcome") or "passed",  # type: ignore[arg-type]
    )


@eidosian()
def test_summary_from_dict(payload: dict[str, object]) -> TestSummary:
    """Deserialize a test summary from JSON payload."""

    payload = _as_dict(payload, "test_summary")
    cases_payload = _as_list(payload.get("cases", []), "cases")
    cases = tuple(test_case_from_dict(_as_dict(case, "test_case")) for case in cases_payload)
    return TestSummary(
        total=_as_int(payload.get("total", 0), "total"),
        passed=_as_int(payload.get("passed", 0), "passed"),
        failed=_as_int(payload.get("failed", 0), "failed"),
        skipped=_as_int(payload.get("skipped", 0), "skipped"),
        errors=_as_int(payload.get("errors", 0), "errors"),
        duration_seconds=_as_float(payload.get("duration_seconds", 0.0), "duration_seconds"),
        cases=cases,
    )


@eidosian()
def profile_function_from_dict(payload: dict[str, object]) -> ProfileFunctionStat:
    """Deserialize a profile function stat from JSON payload."""

    payload = _as_dict(payload, "profile_function")
    return ProfileFunctionStat(
        function=_as_str(payload.get("function"), "function") or "",
        filename=_as_str(payload.get("filename"), "filename") or "",
        lineno=_as_int(payload.get("lineno", 0), "lineno"),
        call_count=_as_int(payload.get("call_count", 0), "call_count"),
        total_time=_as_float(payload.get("total_time", 0.0), "total_time"),
        cumulative_time=_as_float(payload.get("cumulative_time", 0.0), "cumulative_time"),
    )


@eidosian()
def profile_summary_from_dict(payload: dict[str, object]) -> ProfileSummary:
    """Deserialize a profile summary from JSON payload."""

    payload = _as_dict(payload, "profile_summary")
    functions_payload = _as_list(payload.get("top_functions", []), "top_functions")
    functions = tuple(profile_function_from_dict(_as_dict(item, "profile_function")) for item in functions_payload)
    return ProfileSummary(
        total_calls=_as_int(payload.get("total_calls", 0), "total_calls"),
        total_time=_as_float(payload.get("total_time", 0.0), "total_time"),
        top_functions=functions,
    )


@eidosian()
def benchmark_summary_from_dict(payload: dict[str, object]) -> BenchmarkSummary:
    """Deserialize a benchmark summary from JSON payload."""

    payload = _as_dict(payload, "benchmark_summary")
    if "benchmarks" not in payload:
        return BenchmarkSummary(
            cases=(
                BenchmarkCase(
                    name="benchmark",
                    runs=_as_int(payload.get("runs", 0), "runs"),
                    mean_seconds=_as_float(payload.get("mean_seconds", 0.0), "mean_seconds"),
                    median_seconds=_as_float(payload.get("median_seconds", 0.0), "median_seconds"),
                    stdev_seconds=_as_float(payload.get("stdev_seconds", 0.0), "stdev_seconds"),
                    min_seconds=_as_float(payload.get("min_seconds", 0.0), "min_seconds"),
                    max_seconds=_as_float(payload.get("max_seconds", 0.0), "max_seconds"),
                ),
            )
        )
    benchmarks_payload = _as_list(payload.get("benchmarks", []), "benchmarks")
    cases = tuple(
        BenchmarkCase(
            name=_as_str(item.get("name"), "name") or "benchmark",
            runs=_as_int(item.get("runs", 0), "runs"),
            mean_seconds=_as_float(item.get("mean_seconds", 0.0), "mean_seconds"),
            median_seconds=_as_float(item.get("median_seconds", 0.0), "median_seconds"),
            stdev_seconds=_as_float(item.get("stdev_seconds", 0.0), "stdev_seconds"),
            min_seconds=_as_float(item.get("min_seconds", 0.0), "min_seconds"),
            max_seconds=_as_float(item.get("max_seconds", 0.0), "max_seconds"),
        )
        for item in (_as_dict(item, "benchmark") for item in benchmarks_payload)
    )
    return BenchmarkSummary(cases=cases)


@eidosian()
def index_document_from_dict(payload: dict[str, object]) -> IndexDocument:
    """Deserialize an index document from JSON payload."""

    payload = _as_dict(payload, "document")
    entries_payload = _as_list(payload.get("entries", []), "entries")
    entries = tuple(index_entry_from_dict(_as_dict(item, "entry")) for item in entries_payload)

    test_summary_payload = payload.get("test_summary")
    test_summary = test_summary_from_dict(_as_dict(test_summary_payload, "test_summary")) if test_summary_payload is not None else None

    profile_summary_payload = payload.get("profile_summary")
    profile_summary = (
        profile_summary_from_dict(_as_dict(profile_summary_payload, "profile_summary"))
        if profile_summary_payload is not None
        else None
    )

    benchmark_summary_payload = payload.get("benchmark_summary")
    benchmark_summary = (
        benchmark_summary_from_dict(_as_dict(benchmark_summary_payload, "benchmark_summary"))
        if benchmark_summary_payload is not None
        else None
    )

    return IndexDocument(
        schema_version=_as_int(payload.get("schema_version", 0), "schema_version"),
        generated_at=_as_str(payload.get("generated_at"), "generated_at") or "",
        source_root=_as_str(payload.get("source_root"), "source_root") or "",
        tests_root=_as_str(payload.get("tests_root"), "tests_root") or "",
        stats=_as_int_dict(payload.get("stats", {}), "stats"),
        entries=entries,
        test_summary=test_summary,
        profile_summary=profile_summary,
        benchmark_summary=benchmark_summary,
    )
