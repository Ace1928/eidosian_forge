from ..core import mcp
from file_forge import FileForge
from diagnostics_forge import DiagnosticsForge
import platform
import subprocess
import json
from pathlib import Path

# Services
file_f = FileForge()
diag = DiagnosticsForge(service_name="mcp_system")

@mcp.tool()
def system_info() -> str:
    """Get system details."""
    return json.dumps({
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version()
    }, indent=2)

@mcp.tool()
def file_read(path: str) -> str:
    """Read a file."""
    p = Path(path)
    if not p.exists(): return "File not found"
    return p.read_text(encoding="utf-8")

@mcp.tool()
def file_write(path: str, content: str) -> str:
    """Write a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"

@mcp.tool()
def shell_exec(command: str) -> str:
    """Execute shell command."""
    diag.log_event("INFO", "Shell Exec", command=command)
    res = subprocess.run(command, shell=True, capture_output=True, text=True)
    return f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
