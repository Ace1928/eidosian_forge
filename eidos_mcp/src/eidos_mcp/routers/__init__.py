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
from . import consciousness  # noqa: E402,F401
from . import diagnostics  # noqa: E402,F401
from . import gis  # noqa: E402,F401
from . import llm  # noqa: E402,F401
from . import knowledge  # noqa: E402,F401
from . import memory  # noqa: E402,F401
from . import moltbook  # noqa: E402,F401
from . import nexus  # noqa: E402,F401
from . import refactor  # noqa: E402,F401
from . import sms  # noqa: E402,F401
from . import system  # noqa: E402,F401
from . import tika  # noqa: E402,F401
from . import tiered_memory  # noqa: E402,F401
from . import types  # noqa: E402,F401
from . import word_forge  # noqa: E402,F401
from . import erais  # noqa: E402,F401
