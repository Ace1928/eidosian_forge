from __future__ import annotations

# Router modules register tools on import.
from ..forge_loader import prepare_forge_imports

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

from . import audit  # noqa: E402,F401
from . import diagnostics  # noqa: E402,F401
from . import knowledge  # noqa: E402,F401
from . import memory  # noqa: E402,F401
from . import refactor  # noqa: E402,F401
from . import system  # noqa: E402,F401
from . import types  # noqa: E402,F401
