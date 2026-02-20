from __future__ import annotations

import json
from pathlib import Path

from eidosian_core import eidosian

from ..core import tool
from ..forge_loader import ensure_forge_import

ensure_forge_import("refactor_forge")

try:
    from refactor_forge.analyzer import CodeAnalyzer
except Exception:  # pragma: no cover - optional dependency
    CodeAnalyzer = None


from ..state import FORGE_DIR, ROOT_DIR, refactor


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
    if not refactor or not getattr(refactor, "CodeAnalyzer", None):
        # Fallback manual import if needed or check if CodeAnalyzer is available
        try:
            from refactor_forge.analyzer import CodeAnalyzer
        except ImportError:
            return "Error: Refactor forge unavailable"
    else:
        CodeAnalyzer = refactor.CodeAnalyzer

    # Intelligent path resolution
    candidates = [
        Path(path).expanduser().resolve(),  # Absolute or relative to CWD
        ROOT_DIR / path,  # Relative to system root
        FORGE_DIR / path,  # Relative to forge root
    ]

    source_path = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            source_path = candidate
            break

    if not source_path:
        return f"Error: File not found. Checked: {[str(c) for c in candidates]}"

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
