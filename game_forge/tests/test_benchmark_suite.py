from __future__ import annotations

import json
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


def test_benchmark_suite_summary_dry_run(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "game_forge" / "tools" / "benchmark_suite.py"
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "policy_manifest.json"
    manifest_payload = {
        "version": "1.0",
        "allowed_tools": ["game_forge/tools/run.py"],
        "io": {"read_paths": ["game_forge/"], "write_paths": ["game_forge/tools/artifacts/"]},
        "network": {"allowed": False},
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--dry-run",
            "--no-check-deps",
            "--only",
            "agentic-chess",
            "--summary",
            str(summary_path),
            "--policy-manifest",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["benchmarks"][0]["status"] == "dry-run"
    assert payload["policy_manifest_path"] == str(manifest_path)
    assert payload["policy_manifest"]["version"] == "1.0"
