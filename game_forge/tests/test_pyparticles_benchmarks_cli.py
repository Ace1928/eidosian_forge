from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_pyparticles_engine_benchmark_output(tmp_path: Path) -> None:
    output = tmp_path / "pyparticles_engine.json"
    result = subprocess.run(
        [
            sys.executable,
            "game_forge/pyparticles/benchmarks/benchmark.py",
            "--particles",
            "64",
            "--steps",
            "3",
            "--dt",
            "0.01",
            "--warmup",
            "0",
            "--no-profile",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Benchmarking Physics Engine" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["particles"] == 64
    assert payload["steps"] == 3


def test_pyparticles_sim_benchmark_output(tmp_path: Path) -> None:
    output = tmp_path / "pyparticles_sim.json"
    result = subprocess.run(
        [
            sys.executable,
            "game_forge/pyparticles/benchmarks/benchmark_sim.py",
            "--particles",
            "64",
            "--steps",
            "3",
            "--dt",
            "0.01",
            "--warmup",
            "0",
            "--no-profile",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Benchmarking with N=64 particles" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["particles"] == 64
    assert payload["steps"] == 3
