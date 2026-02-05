from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _env_with_paths() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    game_src = root / "game_forge" / "src"
    agent_src = root / "agent_forge" / "src"
    lib_dir = root / "lib"
    env = dict(os.environ)
    pythonpath = os.pathsep.join(
        [str(game_src), str(agent_src), str(lib_dir), env.get("PYTHONPATH", "")]
    )
    env["PYTHONPATH"] = pythonpath
    return env


def test_agentic_chess_list_agents() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agentic_chess", "--list-agents"],
        check=False,
        capture_output=True,
        text=True,
        env=_env_with_paths(),
    )

    assert result.returncode == 0
    assert "available agents" in result.stdout.lower()
