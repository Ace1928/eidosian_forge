#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from eidosian_runtime import write_runtime_capabilities


def main() -> int:
    payload = write_runtime_capabilities()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
