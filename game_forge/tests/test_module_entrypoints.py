from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _env_with_paths() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    game_src = root / "game_forge" / "src"
    lib_dir = root / "lib"
    env = dict(os.environ)
    pythonpath = os.pathsep.join([str(game_src), str(lib_dir), env.get("PYTHONPATH", "")])
    env["PYTHONPATH"] = pythonpath
    return env


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        check=False,
        capture_output=True,
        text=True,
        env=_env_with_paths(),
    )


def test_algorithms_lab_entrypoint_list() -> None:
    result = _run_module("algorithms_lab", "--list")
    assert result.returncode == 0
    assert "available actions" in result.stdout.lower()


def test_chess_game_entrypoint_list() -> None:
    result = _run_module("chess_game", "--list")
    assert result.returncode == 0
    assert "available variants" in result.stdout.lower()


def test_snake_ai_legacy_entrypoint_list() -> None:
    result = _run_module("snake_ai_legacy", "--list")
    assert result.returncode == 0
    assert "available variants" in result.stdout.lower()


def test_stratum_entrypoint_list() -> None:
    result = _run_module("Stratum", "--list")
    assert result.returncode == 0
    assert "collapse" in result.stdout.lower()


def test_agentic_chess_entrypoint_list() -> None:
    result = _run_module("agentic_chess", "--list-agents")
    assert result.returncode == 0
    assert "available agents" in result.stdout.lower()
