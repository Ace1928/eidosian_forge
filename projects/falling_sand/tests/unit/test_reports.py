import cProfile
import json
from pathlib import Path

import pytest

from falling_sand import reports


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_read_junit_reports(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    _write_file(
        report,
        """
<testsuite tests="2" failures="1" errors="0" skipped="0" time="0.5">
  <testcase classname="demo" name="test_ok" time="0.1" />
  <testcase classname="demo" name="test_fail" time="0.4">
    <failure message="oops" />
  </testcase>
</testsuite>
""".strip(),
    )

    summary = reports.read_junit_reports([report])

    assert summary.total == 2
    assert summary.failed == 1
    assert summary.passed == 1
    assert summary.duration_seconds == 0.5
    assert len(summary.cases) == 2


def test_read_profile_stats(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.pstats"

    def sample() -> int:
        return sum(range(10))

    profiler = cProfile.Profile()
    profiler.runcall(sample)
    profiler.dump_stats(str(profile_path))

    summary = reports.read_profile_stats(profile_path, top_n=5)

    assert summary.total_calls > 0
    assert summary.total_time >= 0
    assert summary.top_functions


def test_read_profile_stats_invalid_top_n(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.pstats"

    def sample() -> int:
        return sum(range(5))

    profiler = cProfile.Profile()
    profiler.runcall(sample)
    profiler.dump_stats(str(profile_path))

    with pytest.raises(ValueError, match="top_n must be positive"):
        reports.read_profile_stats(profile_path, top_n=0)


def test_read_benchmark_report(tmp_path: Path) -> None:
    report = tmp_path / "benchmark.json"
    payload = {
        "benchmarks": [
            {
                "name": "indexer",
                "runs": 3,
                "samples": [0.01, 0.02, 0.03],
            }
        ]
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    summary = reports.read_benchmark_report(report)

    assert summary.cases[0].runs == 3
    assert summary.cases[0].min_seconds == 0.01
    assert summary.cases[0].max_seconds == 0.03


def test_read_benchmark_report_mismatch(tmp_path: Path) -> None:
    report = tmp_path / "benchmark.json"
    payload = {
        "benchmarks": [
            {
                "name": "indexer",
                "runs": 2,
                "samples": [0.01, 0.02, 0.03],
            }
        ]
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="run count does not match samples"):
        reports.read_benchmark_report(report)


def test_read_junit_reports_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JUnit report not found"):
        reports.read_junit_reports([tmp_path / "missing.xml"])
