"""Module entrypoint for ECosmos."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_imports() -> None:
    package_dir = Path(__file__).resolve().parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))


def main() -> int:
    _ensure_local_imports()
    from .main import main as run_main

    result = run_main()
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
