from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_launcher_list() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "game_forge" / "tools" / "run.py"

    result = subprocess.run(
        [sys.executable, str(runner), "--list"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "gene-particles" in result.stdout
    assert "ecosmos" in result.stdout
    assert "algorithms-lab-benchmark" in result.stdout
    assert "algorithms-lab-profiler" in result.stdout
    assert "agentic-chess" in result.stdout
    assert "agentic-chess-benchmark" in result.stdout
    assert "falling-sand-benchmark" in result.stdout
    assert "falling-sand-profile-index" in result.stdout
    assert "falling-sand-index" in result.stdout
    assert "falling-sand-ingest" in result.stdout
    assert "falling-sand-report" in result.stdout
    assert "stratum-benchmark" in result.stdout
    assert "pyparticles-benchmark" in result.stdout
    assert "pyparticles-benchmark-sim" in result.stdout
