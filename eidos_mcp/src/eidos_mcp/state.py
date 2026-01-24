from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Optional, Type

from .forge_loader import ensure_forge_import
from eidosian_core import eidosian


FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge")).resolve()
ROOT_DIR = Path(os.environ.get("EIDOS_ROOT_DIR", "/home/lloyd")).resolve()


def _import_symbol(module_name: str, symbol: str) -> Optional[Type[Any]]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, symbol, None)


ensure_forge_import("gis_forge")
GisCore = _import_symbol("gis_forge.core", "GisCore")
GIS_PATH = Path(os.environ.get("EIDOS_GIS_PATH", FORGE_DIR / "gis_forge" / "gis_data.json"))
gis = GisCore(persistence_path=GIS_PATH) if GisCore else None

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
