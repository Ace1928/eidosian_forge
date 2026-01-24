from __future__ import annotations

import json

from ..core import tool
from ..forge_loader import ensure_forge_import
from eidosian_core import eidosian

ensure_forge_import("diagnostics_forge")

try:
    from diagnostics_forge.core import DiagnosticsForge
except Exception:  # pragma: no cover - optional dependency
    DiagnosticsForge = None


diag = DiagnosticsForge(service_name="eidos_mcp") if DiagnosticsForge else None


@tool(
    name="diagnostics_ping",
    description="Return basic diagnostics status.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def diagnostics_ping() -> str:
    """Return basic diagnostics status."""
    if not diag:
        return "Error: Diagnostics forge unavailable"
    diag.log_event("INFO", "diagnostics_ping")
    return "ok"


@tool(
    name="diagnostics_metrics",
    description="Return diagnostics metrics summary.",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
    },
)
@eidosian()
def diagnostics_metrics(name: str | None = None) -> str:
    """Return diagnostics metrics summary."""
    if not diag:
        return "Error: Diagnostics forge unavailable"
    if name:
        summary = diag.get_metrics_summary(name)
    else:
        summary = {k: diag.get_metrics_summary(k) for k in diag.metrics}
    return json.dumps(summary, indent=2)
