from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_benchmark_suite_list() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "game_forge" / "tools" / "benchmark_suite.py"

    result = subprocess.run(
        [sys.executable, str(runner), "--list"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "agentic-chess" in result.stdout
    assert "gene-particles" in result.stdout
    assert "algorithms-lab" in result.stdout
    assert "falling-sand" in result.stdout
    assert "stratum" in result.stdout
    assert "pyparticles-engine" in result.stdout
    assert "pyparticles-sim" in result.stdout


def test_benchmark_suite_dry_run() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "game_forge" / "tools" / "benchmark_suite.py"

    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--dry-run",
            "--no-check-deps",
            "--only",
            "agentic-chess",
            "--tag",
            "test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "dry-run" in result.stdout
    assert "agentic-chess-benchmark" in result.stdout
