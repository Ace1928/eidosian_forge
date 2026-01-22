from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Iterable


_FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge")).resolve()


def _ensure_forge_subdir(module_name: str) -> None:
    forge_path = _FORGE_DIR / module_name
    if forge_path.exists():
        src_path = forge_path / "src"
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))


def ensure_forge_import(module_name: str) -> bool:
    """Best-effort import for a forge module, ensuring the forge paths exist."""
    _ensure_forge_subdir(module_name)
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        try:
            importlib.import_module(module_name)
            return True
        except Exception:
            return False


def prepare_forge_imports(modules: Iterable[str]) -> None:
    """Warm imports for a list of forges without hard-failing."""
    for module_name in modules:
        ensure_forge_import(module_name)
