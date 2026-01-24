from __future__ import annotations

import json
from pathlib import Path

from ..core import tool
from ..forge_loader import ensure_forge_import
from eidosian_core import eidosian

ensure_forge_import("refactor_forge")

try:
    from refactor_forge.analyzer import CodeAnalyzer
except Exception:  # pragma: no cover - optional dependency
    CodeAnalyzer = None


@tool(
    name="refactor_analyze",
    description="Analyze a Python file for structural boundaries and dependencies.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)
@eidosian()
def refactor_analyze(path: str) -> str:
    """Analyze a Python file for structural boundaries and dependencies."""
    if not CodeAnalyzer:
        return "Error: Refactor forge unavailable"
    source_path = Path(path).expanduser().resolve()
    analysis = CodeAnalyzer(source_path).analyze()
    deps = []
    if hasattr(analysis.get("dependencies"), "edges"):
        deps = [{"from": u, "to": v} for u, v in analysis["dependencies"].edges()]
    payload = {
        "file_info": analysis.get("file_info"),
        "modules": analysis.get("modules"),
        "dependencies": deps,
    }
    return json.dumps(payload, indent=2)
