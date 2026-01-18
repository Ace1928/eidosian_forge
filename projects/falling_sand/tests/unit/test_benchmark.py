import json
from pathlib import Path

import pytest

from scripts import benchmark_index


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_benchmark_index(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    tests_root = tmp_path / "tests"
    _write_file(source_root / "pkg.py", "def demo():\n    return 1\n")
    _write_file(tests_root / "test_pkg.py", "def test_demo():\n    assert True\n")

    samples = benchmark_index.benchmark_index(source_root, tests_root, runs=2)

    assert len(samples) == 2
    assert all(sample >= 0 for sample in samples)


def test_benchmark_index_invalid_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runs must be positive"):
        benchmark_index.benchmark_index(tmp_path, tmp_path, runs=0)


def test_write_benchmark_report(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    benchmark_index.write_benchmark_report([0.1, 0.2], output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["runs"] == 2
    assert payload["min_seconds"] == 0.1
    assert payload["max_seconds"] == 0.2


def test_main(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    tests_root = tmp_path / "tests"
    _write_file(source_root / "pkg.py", "def demo():\n    return 1\n")
    _write_file(tests_root / "test_pkg.py", "def test_demo():\n    assert True\n")

    output = tmp_path / "benchmark.json"
    exit_code = benchmark_index.main(
        [
            "--source-root",
            str(source_root),
            "--tests-root",
            str(tests_root),
            "--runs",
            "2",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.exists()


def test_benchmark_report_backward_compat(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    benchmark_index.write_benchmark_report([0.1, 0.2], output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["runs"] == 2
