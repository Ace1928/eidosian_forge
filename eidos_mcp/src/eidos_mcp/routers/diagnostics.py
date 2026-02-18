from __future__ import annotations
import json
from ..core import tool
from diagnostics_forge.core import DiagnosticsForge
from eidosian_core import eidosian

# Initialize core diagnostics
diag = DiagnosticsForge()

@tool(
    name="diagnostics_pulse",
    description="Get real-time system resource usage (CPU, RAM, Disk).",
)
@eidosian()
def diagnostics_pulse() -> str:
    """Get real-time system resource usage."""
    pulse = diag.get_system_pulse()
    return json.dumps(pulse, indent=2)

@tool(
    name="diagnostics_processes",
    description="List active Eidosian-related processes and their resource consumption.",
)
@eidosian()
def diagnostics_processes() -> str:
    """List active Eidosian-related processes."""
    procs = diag.get_process_metrics()
    return json.dumps(procs, indent=2)
