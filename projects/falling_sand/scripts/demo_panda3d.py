"""Run the Panda3D falling-sand demo."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from falling_sand.engine.demo import run_demo
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from falling_sand.engine.demo import run_demo  # type: ignore[import-not-found]


if __name__ == "__main__":
    run_demo()
