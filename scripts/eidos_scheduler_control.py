#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parent.parent
for extra in (FORGE_ROOT / "lib", FORGE_ROOT / "scripts", FORGE_ROOT):
    value = str(extra)
    if extra.exists() and value not in sys.path:
        sys.path.insert(0, value)

from eidos_scheduler import apply_scheduler_control


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    action = args[0] if args else "status"
    payload = apply_scheduler_control(action)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
