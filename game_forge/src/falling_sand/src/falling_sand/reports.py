"""Report parsing for tests, profiling, and benchmarks."""

from __future__ import annotations

import json
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import pstats

from falling_sand.models import (
    BenchmarkCase,
    BenchmarkSummary,
    ProfileFunctionStat,
    ProfileSummary,
    TestCaseResult,
    TestOutcome,
    TestSummary,
)
from eidosian_core import eidosian


@dataclass(frozen=True)
class ParsedTestSuite:
    """Internal container for raw test suite values."""

    total: int
    failures: int
    errors: int
    skipped: int
    duration_seconds: float
    cases: tuple[TestCaseResult, ...]


def _parse_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("Invalid float value: boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid float value: {value}") from exc
    raise ValueError(f"Invalid float value: {value}")


def _parse_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("Invalid integer value: boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid integer value: {value}") from exc
    raise ValueError(f"Invalid integer value: {value}")


def _iter_test_suites(root: ET.Element) -> Iterable[ET.Element]:
    if root.tag == "testsuite":
        yield root
    elif root.tag == "testsuites":
        yield from root.iter("testsuite")
    else:
        raise ValueError(f"Unsupported JUnit root element: {root.tag}")


def _parse_testcase(node: ET.Element) -> TestCaseResult:
    name = node.attrib.get("name") or ""
    classname = node.attrib.get("classname")
    file_attr = node.attrib.get("file")
    line_attr = node.attrib.get("line")
    duration_seconds = _parse_float(node.attrib.get("time"), 0.0)

    outcome: TestOutcome = "passed"
    if node.find("failure") is not None:
        outcome = "failed"
    elif node.find("error") is not None:
        outcome = "error"
    elif node.find("skipped") is not None:
        outcome = "skipped"

    line = None
    if line_attr is not None:
        line = _parse_int(line_attr)
        if line <= 0:
            raise ValueError("JUnit line attribute must be positive")

    return TestCaseResult(
        name=name,
        classname=classname,
        file=file_attr,
        line=line,
        duration_seconds=duration_seconds,
        outcome=outcome,
    )


def _parse_suite(root: ET.Element) -> ParsedTestSuite:
    total = _parse_int(root.attrib.get("tests"), 0)
    failures = _parse_int(root.attrib.get("failures"), 0)
    errors = _parse_int(root.attrib.get("errors"), 0)
    skipped = _parse_int(root.attrib.get("skipped"), 0)
    duration_seconds = _parse_float(root.attrib.get("time"), 0.0)

    cases = tuple(_parse_testcase(node) for node in root.iter("testcase"))
    return ParsedTestSuite(
        total=total,
        failures=failures,
        errors=errors,
        skipped=skipped,
        duration_seconds=duration_seconds,
        cases=cases,
    )


@eidosian()
def read_junit_reports(paths: Iterable[Path]) -> TestSummary:
    """Parse one or more JUnit XML reports into a summary."""

    suites: list[ParsedTestSuite] = []
    for path in paths:
        if not path.exists():
            raise ValueError(f"JUnit report not found: {path}")
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid JUnit XML: {path}") from exc
        for suite in _iter_test_suites(tree.getroot()):
            suites.append(_parse_suite(suite))

    if not suites:
        raise ValueError("No JUnit test suites found")

    total = sum(suite.total for suite in suites)
    failures = sum(suite.failures for suite in suites)
    errors = sum(suite.errors for suite in suites)
    skipped = sum(suite.skipped for suite in suites)
    duration_seconds = sum(suite.duration_seconds for suite in suites)
    cases = tuple(case for suite in suites for case in suite.cases)
    passed = total - failures - errors - skipped

    return TestSummary(
        total=total,
        passed=passed,
        failed=failures,
        skipped=skipped,
        errors=errors,
        duration_seconds=duration_seconds,
        cases=cases,
    )


@eidosian()
def read_profile_stats(path: Path, top_n: int = 20) -> ProfileSummary:
    """Parse a cProfile stats file into a summary."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not path.exists():
        raise ValueError(f"Profile stats not found: {path}")
    stats = cast(Any, pstats.Stats(str(path)))
    total_calls = sum(value[1] for value in stats.stats.values())
    total_time = stats.total_tt

    functions: list[ProfileFunctionStat] = []
    for (filename, lineno, funcname), values in stats.stats.items():
        primitive_calls, call_count, total_time_fn, cumulative_time, _ = values
        functions.append(
            ProfileFunctionStat(
                function=funcname,
                filename=filename,
                lineno=lineno,
                call_count=call_count,
                total_time=total_time_fn,
                cumulative_time=cumulative_time,
            )
        )

    functions.sort(key=lambda item: item.cumulative_time, reverse=True)
    top_functions = tuple(functions[:top_n])

    return ProfileSummary(
        total_calls=total_calls,
        total_time=total_time,
        top_functions=top_functions,
    )


@eidosian()
def read_benchmark_report(path: Path) -> BenchmarkSummary:
    """Parse a benchmark JSON report into a summary."""

    if not path.exists():
        raise ValueError(f"Benchmark report not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid benchmark JSON: {path}") from exc

    if "benchmarks" not in payload:
        return _read_legacy_benchmark(payload)
    benchmarks = payload.get("benchmarks", [])
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("Benchmark report missing benchmarks")
    cases = tuple(_parse_benchmark_case(item) for item in benchmarks)
    return BenchmarkSummary(cases=cases)


def _read_legacy_benchmark(payload: dict[str, object]) -> BenchmarkSummary:
    runs = _parse_int(payload.get("runs", 0), 0)
    samples = _coerce_samples(payload.get("samples", []))
    if runs <= 0 or not samples:
        raise ValueError("Benchmark report missing samples")
    if runs != len(samples):
        raise ValueError("Benchmark report run count does not match samples")

    mean_seconds = _parse_float(payload.get("mean_seconds", statistics.mean(samples)), 0.0)
    median_seconds = _parse_float(
        payload.get("median_seconds", statistics.median(samples)), 0.0
    )
    stdev_seconds = _parse_float(
        payload.get("stdev_seconds", statistics.pstdev(samples)), 0.0
    )
    min_seconds = _parse_float(payload.get("min_seconds", min(samples)), 0.0)
    max_seconds = _parse_float(payload.get("max_seconds", max(samples)), 0.0)

    return BenchmarkSummary(
        cases=(
            BenchmarkCase(
                name="benchmark",
                runs=runs,
                mean_seconds=mean_seconds,
                median_seconds=median_seconds,
                stdev_seconds=stdev_seconds,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
            ),
        )
    )


def _parse_benchmark_case(payload: object) -> BenchmarkCase:
    if not isinstance(payload, dict):
        raise ValueError("Benchmark entry must be an object")
    name = payload.get("name") or "benchmark"
    runs = _parse_int(payload.get("runs", 0), 0)
    samples = _coerce_samples(payload.get("samples", []))
    if runs <= 0 or not samples:
        raise ValueError("Benchmark entry missing samples")
    if runs != len(samples):
        raise ValueError("Benchmark entry run count does not match samples")

    mean_seconds = _parse_float(payload.get("mean_seconds", statistics.mean(samples)), 0.0)
    median_seconds = _parse_float(
        payload.get("median_seconds", statistics.median(samples)), 0.0
    )
    stdev_seconds = _parse_float(
        payload.get("stdev_seconds", statistics.pstdev(samples)), 0.0
    )
    min_seconds = _parse_float(payload.get("min_seconds", min(samples)), 0.0)
    max_seconds = _parse_float(payload.get("max_seconds", max(samples)), 0.0)

    return BenchmarkCase(
        name=str(name),
        runs=runs,
        mean_seconds=mean_seconds,
        median_seconds=median_seconds,
        stdev_seconds=stdev_seconds,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )


def _coerce_samples(value: object) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("Benchmark samples must be a list")
    samples: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("Benchmark sample must be a number")
        if isinstance(item, (int, float)):
            samples.append(float(item))
            continue
        raise ValueError("Benchmark sample must be a number")
    return samples
