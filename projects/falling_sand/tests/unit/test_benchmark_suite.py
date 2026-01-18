import json
from pathlib import Path

import pytest

from falling_sand import benchmarks


def test_benchmark_suite_outputs(tmp_path: Path) -> None:
    output = tmp_path / "bench.json"
    exit_code = benchmarks.main([
        "--runs",
        "2",
        "--output",
        str(output),
    ])

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "benchmarks" in payload
    assert {bench["name"] for bench in payload["benchmarks"]} == {
        "indexer",
        "simulation",
        "streaming",
        "terrain",
    }


def test_benchmark_suite_invalid_runs() -> None:
    with pytest.raises(ValueError, match="runs must be positive"):
        benchmarks.benchmark_indexer(Path("src"), Path("tests"), runs=0)
    with pytest.raises(ValueError, match="runs must be positive"):
        benchmarks.benchmark_streaming(runs=0)
    with pytest.raises(ValueError, match="runs must be positive"):
        benchmarks.benchmark_terrain_generation(runs=0)


def test_benchmark_terrain_generation() -> None:
    samples = benchmarks.benchmark_terrain_generation(runs=2, size=4)
    assert len(samples) == 2
    assert all(sample >= 0 for sample in samples)
