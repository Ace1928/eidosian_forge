from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Type

from eidosian_core import eidosian

from . import FORGE_ROOT
from .forge_loader import ensure_forge_import

FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", str(FORGE_ROOT))).resolve()
ROOT_DIR = Path(os.environ.get("EIDOS_ROOT_DIR", str(FORGE_DIR.parent))).resolve()


def _import_symbol(module_name: str, symbol: str) -> Optional[Type[Any]]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        # Retry once after forcing forge path setup and clearing namespace-only stubs.
        root_module = module_name.split(".", 1)[0]
        ensure_forge_import(root_module)
        existing = sys.modules.get(root_module)
        if existing is not None and getattr(existing, "__file__", None) is None:
            sys.modules.pop(root_module, None)
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return None
    return getattr(module, symbol, None)


ensure_forge_import("gis_forge")
GisCore = _import_symbol("gis_forge.core", "GisCore")
GIS_PATH = Path(os.environ.get("EIDOS_GIS_PATH", FORGE_DIR / "gis_forge" / "gis_data.json"))


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        return bool(first) and first[0].startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def _is_valid_json(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def _init_gis() -> Any:
    if not GisCore:
        return None

    primary = GIS_PATH
    fallback = Path.home() / ".eidosian" / "gis_data.local.json"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    if not fallback.exists():
        fallback.write_text(json.dumps({}, indent=2), encoding="utf-8")

    if _looks_like_lfs_pointer(primary) or not _is_valid_json(primary):
        candidates = [fallback, primary]
    else:
        candidates = [primary, fallback]
    for candidate in candidates:
        try:
            return GisCore(persistence_path=candidate)
        except Exception:
            continue
    return None


gis = _init_gis()

ensure_forge_import("audit_forge")
AuditForge = _import_symbol("audit_forge.audit_core", "AuditForge")
audit = AuditForge(FORGE_DIR / "audit_data") if AuditForge else None

ensure_forge_import("type_forge")
TypeCore = _import_symbol("type_forge.core", "TypeCore")
type_forge = TypeCore() if TypeCore else None

ensure_forge_import("llm_forge")
ModelManager = _import_symbol("llm_forge.core.manager", "ModelManager")


class _StubLLM:
    class _Response:
        def __init__(self, text: str | None = None):
            self.text = text

    @eidosian()
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self._Response(text=None)


if ModelManager:
    try:
        llm = ModelManager()
    except Exception:
        llm = _StubLLM()
else:
    llm = _StubLLM()

ensure_forge_import("agent_forge")
AgentForge = _import_symbol("agent_forge.agent_core", "AgentForge")
agent = AgentForge(llm=llm, require_approval=True) if AgentForge else None

ensure_forge_import("refactor_forge")
try:
    refactor = importlib.import_module("refactor_forge")
except Exception:
    refactor = None
