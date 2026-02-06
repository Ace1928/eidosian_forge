#!/usr/bin/env python3
"""Profile the Moltbook Nexus UI render path."""

from __future__ import annotations

import cProfile
import os
import pstats
from pathlib import Path

import sys
from fastapi.testclient import TestClient

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MOLTBOOK_MOCK", "true")
os.environ.setdefault("MOLTBOOK_ALLOWED_HOSTS", "testserver")

from moltbook_forge.ui.app import app


def main() -> int:
    out_dir = Path("data/moltbook")
    out_dir.mkdir(parents=True, exist_ok=True)
    prof_path = out_dir / "profile_ui.prof"
    stats_path = out_dir / "profile_ui.txt"

    profiler = cProfile.Profile()
    profiler.enable()
    with TestClient(app) as client:
        response = client.get("/")
        if response.status_code != 200:
            print("ERROR dashboard returned non-200")
            return 1
    profiler.disable()
    profiler.dump_stats(str(prof_path))

    with stats_path.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profiler, stream=handle).sort_stats("cumulative")
        stats.print_stats(50)
    print(f"Wrote profile to {prof_path} and {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
