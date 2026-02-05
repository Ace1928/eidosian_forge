from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_falling_sand_benchmark_index_output(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    result = subprocess.run(
        [
            sys.executable,
            "game_forge/src/falling_sand/scripts/benchmark_index.py",
            "--runs",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["runs"] == 1
