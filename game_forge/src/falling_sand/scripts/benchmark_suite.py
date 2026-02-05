"""Run multiple benchmarks and write a consolidated report."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ is None:
    sys.path.append(str(PROJECT_ROOT / "src"))

from falling_sand.benchmarks import main


if __name__ == "__main__":
    raise SystemExit(main())
