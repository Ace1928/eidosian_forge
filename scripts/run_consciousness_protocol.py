#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_paths(root: Path) -> None:
    extra = [root / "lib", root / "eidos_mcp" / "src", root]
    for path in extra:
        s = str(path)
        if path.exists() and s not in sys.path:
            sys.path.insert(0, s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run operational consciousness assessment and emit JSON report.")
    parser.add_argument("--trials", type=int, default=3, help="Trials per probe (default: 3)")
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not write report to disk",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    _ensure_paths(root)

    from eidos_mcp.consciousness_protocol import ConsciousnessProtocol

    protocol = ConsciousnessProtocol(root_dir=root)
    report = protocol.run_assessment(trials=max(1, args.trials), persist=not args.no_persist)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
