from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_algorithms_lab_profiler_output(tmp_path: Path) -> None:
    output = tmp_path / "algorithms_lab.prof"
    script = Path(__file__).resolve().parents[1] / "tools" / "algorithms_lab" / "profiler.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
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
