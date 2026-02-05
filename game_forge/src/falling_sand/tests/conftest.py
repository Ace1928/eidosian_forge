from __future__ import annotations

import sys
from pathlib import Path


def _prepend_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

_prepend_path(BASE_DIR)
_prepend_path(SRC_DIR)
