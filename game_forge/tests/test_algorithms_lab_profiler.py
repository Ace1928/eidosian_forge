from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_algorithms_lab_profiler_output(tmp_path: Path) -> None:
    output = tmp_path / "algorithms_lab.prof"
    result = subprocess.run(
        [
            sys.executable,
            "game_forge/tools/algorithms_lab/profiler.py",
            "--algorithm",
            "grid",
            "--particles",
            "32",
            "--steps",
            "2",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "INFO profile saved" in result.stdout
    assert output.exists()
