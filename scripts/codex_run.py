#!/usr/bin/env python3
"""Helper script that runs the Codex agent via the launcher shell script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from eidosian_core import eidosian

DEFAULT_QUERY = (
    "go through this file, which may or may not be a repository, check for any existing "
    "documentation, add to it and ensure it is accurate and up to date, if none exists then "
    "create a set of documentation for this folder. If you need to use an agent to assist you, "
    "you can run the codex_run.py file and pass it your query/task to assist you with more "
    "complex/multi-step/multi-purpose tasks."
)


@eidosian()
def main() -> None:
    script_path = Path(__file__).with_name("launch_codex_agent.sh")
    if not script_path.exists():
        raise FileNotFoundError(f"{script_path} is missing; cannot launch the agent.")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUERY
    subprocess.run([str(script_path), query], check=True)


if __name__ == "__main__":
    main()
