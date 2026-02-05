from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_agentic_chess_benchmark_output(tmp_path: Path) -> None:
    pytest.importorskip("chess", reason="python-chess required")
    output = tmp_path / "agentic_chess.json"
    result = subprocess.run(
        [
            sys.executable,
            "game_forge/tools/agentic_chess_benchmark.py",
            "--white",
            "random",
            "--black",
            "random",
            "--games",
            "2",
            "--max-moves",
            "4",
            "--seed",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "agentic_chess benchmark" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["games"] == 2
    assert payload["max_moves"] == 4
