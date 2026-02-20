from __future__ import annotations

import json

from eidosian_core import eidosian

from ..core import tool
from ..forge_loader import ensure_forge_import

ensure_forge_import("diagnostics_forge")

try:
    from diagnostics_forge.core import DiagnosticsForge
except Exception:  # pragma: no cover - optional dependency
    DiagnosticsForge = None

# Initialize diagnostics
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
    # log_event exists in some versions of DiagnosticsForge
    if hasattr(diag, "log_event"):
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
    if not hasattr(diag, "metrics"):
        return json.dumps({"status": "no_metrics_tracking"})

    if name:
        summary = diag.get_metrics_summary(name)
    else:
        summary = {k: diag.get_metrics_summary(k) for k in diag.metrics}
    return json.dumps(summary, indent=2)


@tool(
    name="diagnostics_pulse",
    description="Get real-time system resource usage (CPU, RAM, Disk).",
)
@eidosian()
def diagnostics_pulse() -> str:
    """Get real-time system resource usage."""
    if not diag:
        return "Error: Diagnostics forge unavailable"
    pulse = diag.get_system_pulse()
    return json.dumps(pulse, indent=2)


@tool(
    name="diagnostics_processes",
    description="List active Eidosian-related processes and their resource consumption.",
)
@eidosian()
def diagnostics_processes() -> str:
    """List active Eidosian-related processes."""
    if not diag:
        return "Error: Diagnostics forge unavailable"
    procs = diag.get_process_metrics()
    return json.dumps(procs, indent=2)
