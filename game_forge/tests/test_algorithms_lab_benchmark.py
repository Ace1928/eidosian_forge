from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_algorithms_lab_benchmark_output(tmp_path: Path) -> None:
    output = tmp_path / "algorithms_lab.json"
    script = Path(__file__).resolve().parents[1] / "tools" / "algorithms_lab" / "benchmark.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--algorithms",
            "grid",
            "--particles",
            "16",
            "--steps",
            "2",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "INFO benchmark complete" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["parameters"]["algorithms"] == "grid"
    assert "grid" in payload["results_seconds"]
