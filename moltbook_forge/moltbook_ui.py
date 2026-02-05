#!/usr/bin/env python3
"""Launch the Moltbook UI dashboard.

Example:
  python moltbook_forge/moltbook_ui.py --host 0.0.0.0 --port 8080 --mock
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable


if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Moltbook UI dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--mock", action="store_true", help="Use mock API data")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    os.environ["MOLTBOOK_MOCK"] = "true" if args.mock else "false"
    try:
        import uvicorn
    except ImportError:
        print("ERROR uvicorn is required. Install it in the eidosian_venv.")
        return 1
    uvicorn.run("moltbook_forge.ui.app:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
