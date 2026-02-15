from __future__ import annotations

import os
import sys
from pathlib import Path


def _looks_like_forge_root(path: Path) -> bool:
    return (path / "eidos_mcp").exists() and (path / "lib" / "eidosian_core").exists()


def _detect_forge_root() -> Path:
    candidates: list[Path] = []

    env_root = os.environ.get("EIDOS_FORGE_DIR")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    default_from_package = Path(__file__).resolve().parents[3]
    candidates.append(default_from_package)

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    for candidate in candidates:
        resolved = candidate.resolve()
        if _looks_like_forge_root(resolved):
            return resolved

    return default_from_package.resolve()


FORGE_ROOT = _detect_forge_root()
ROOT_DIR = FORGE_ROOT.parent

os.environ.setdefault("EIDOS_FORGE_DIR", str(FORGE_ROOT))
os.environ.setdefault("EIDOS_ROOT_DIR", str(ROOT_DIR))

for path in (FORGE_ROOT / "lib", FORGE_ROOT):
    if path.exists():
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)
