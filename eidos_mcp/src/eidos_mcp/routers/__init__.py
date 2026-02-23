from __future__ import annotations

from importlib import import_module

from ..forge_loader import prepare_forge_imports
from ..logging_utils import log_debug, log_error

# Router modules register tools on import.
prepare_forge_imports(
    [
        "audit_forge",
        "diagnostics_forge",
        "file_forge",
        "gis_forge",
        "knowledge_forge",
        "llm_forge",
        "memory_forge",
        "refactor_forge",
        "type_forge",
    ]
)

_ROUTER_MODULES = (
    "audit",
    "auth",
    "consciousness",
    "diagnostics",
    "gis",
    "llm",
    "knowledge",
    "learner",
    "memory",
    "moltbook",
    "nexus",
    "prompt",
    "refactor",
    "sms",
    "system",
    "tika",
    "tiered_memory",
    "types",
    "word_forge",
    "erais",
    "plugins",
)


def _safe_import_router(module_name: str) -> None:
    try:
        import_module(f"{__name__}.{module_name}")
    except Exception as exc:  # pragma: no cover - defensive router loading
        log_debug(f"Router import skipped: {module_name} ({exc})")
        log_error(f"router_import:{module_name}", str(exc))


for _module_name in _ROUTER_MODULES:
    _safe_import_router(_module_name)
