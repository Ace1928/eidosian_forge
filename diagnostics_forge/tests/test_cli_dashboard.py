from __future__ import annotations

import json
from pathlib import Path

from diagnostics_forge.cli import main


def test_dashboard_renders_metrics_table(tmp_path: Path, capsys) -> None:
    payload = {
        "step_a": {"count": 2, "avg": 0.2, "min": 0.1, "max": 0.3},
        "step_b": {"count": 1, "avg": 0.9, "min": 0.9, "max": 0.9},
    }
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps(payload), encoding="utf-8")

    code = main(["dashboard", "--metrics-file", str(metrics_file), "--top", "1"])
    out = capsys.readouterr().out

    assert code == 0
    assert "Diagnostics Metrics Dashboard" in out
    assert "step_b" in out
    assert "step_a" not in out


def test_dashboard_missing_file_returns_error(capsys) -> None:
    code = main(["dashboard", "--metrics-file", "/tmp/missing-metrics.json"])
    out = capsys.readouterr().out
    assert code == 1
    assert "metrics file not found" in out
