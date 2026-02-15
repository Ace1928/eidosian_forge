from __future__ import annotations

import os
import subprocess
import json
import time
import sys
from pathlib import Path
from typing import Optional

from ..core import tool
from .. import FORGE_ROOT
from ..state import agent
from eidosian_core import eidosian


@tool(
    name="agent_run_task",
    description="Delegate a complex objective to the Agent Forge. Returns a plan or execution summary.",
    parameters={
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "execute": {"type": "boolean", "description": "If true, execute the plan immediately."}
        },
        "required": ["objective"],
    },
)
@eidosian()
def agent_run_task(objective: str, execute: bool = False) -> str:
    """Delegate a complex objective to the Agent Forge."""
    if not agent:
        return json.dumps({"error": "Agent Forge not available (import failed)"})
    
    goal = agent.create_goal(objective, plan=True)
    
    execution_results = []
    if execute:
        for t in goal.tasks:
            success = agent.execute_task(t)
            execution_results.append({
                "task": t.description,
                "tool": t.tool,
                "status": t.status,
                "result": str(t.result)[:200] + "..." if t.result and len(str(t.result)) > 200 else t.result
            })
            if not success and t.priority > 0: # Stop on critical failure
                 break
                 
        return json.dumps({
            "objective": objective,
            "status": "executed",
            "results": execution_results
        }, indent=2)

    # We return the plan for inspection. 
    # In a full autonomous loop, we might execute it, but for MCP we prefer planning first.
    tasks = []
    for t in goal.tasks:
        tasks.append({
            "description": t.description,
            "tool": t.tool,
            "args": t.kwargs,
            "status": t.status
        })
    
    return json.dumps({
        "objective": objective,
        "plan_id": str(id(goal)),
        "tasks": tasks
    }, indent=2)


@tool(
    name="mcp_self_upgrade",
    description="Upgrade the MCP server (restart service) after verifying tests. Requires a stated benefit.",
    parameters={
        "type": "object",
        "properties": {
            "benefit": {"type": "string"},
            "run_tests": {"type": "boolean"},
        },
        "required": ["benefit"],
    },
)
@eidosian()
def mcp_self_upgrade(benefit: str, run_tests: bool = True) -> str:
    """
    Restart the MCP service to pick up latest code changes.
    Enforces test passing before restart.
    """
    enabled = os.environ.get("EIDOS_MCP_SELF_UPGRADE", "true").lower() == "true"
    if not enabled:
        return "Error: mcp_self_upgrade disabled by configuration."

    if not benefit.strip():
        return "Error: A specific, measurable benefit must be provided for the upgrade."

    forge_dir = os.environ.get("EIDOS_FORGE_DIR", str(FORGE_ROOT))
    
    if run_tests:
        # Run tests in the forge directory
        try:
            # We use the venv python to run pytest
            venv_python = os.environ.get(
                "EIDOS_PYTHON_BIN",
                str(Path(forge_dir) / "eidosian_venv" / "bin" / "python3"),
            )
            if not Path(venv_python).exists():
                venv_python = sys.executable
            result = subprocess.run(
                [venv_python, "-m", "pytest", "eidos_mcp"],
                cwd=forge_dir,
                capture_output=True,
                text=True,
                timeout=300, # 5 minutes max for tests
            )
            if result.returncode != 0:
                return f"Error: Tests failed. Upgrade aborted.\n\nStdout:\n{result.stdout}\n\nStderr:\n{result.stderr}"
        except Exception as e:
            return f"Error: Failed to run tests: {e}"

    # If we got here, tests passed or were skipped. Restart the service.
    # Use explicit user bus in case the MCP process runs without it in env.
    service_name = os.environ.get("EIDOS_MCP_SYSTEMD_SERVICE", "eidos-mcp.service")
    user_bus = os.environ.get("DBUS_SESSION_BUS_ADDRESS") or f"unix:path=/run/user/{os.getuid()}/bus"
    restart_env = dict(os.environ)
    restart_env["DBUS_SESSION_BUS_ADDRESS"] = user_bus

    # Return success before the process potentially restarts itself.
    # We schedule restart slightly in the future so this response can flush cleanly.
    try:
        subprocess.Popen(
            [
                "bash",
                "-lc",
                f"sleep 1 && systemctl --user restart {service_name}",
            ],
            env=restart_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        return f"Error: Failed to schedule service restart: {e}"

    # Best-effort wait to verify the command can at least resolve service state.
    # This does not block for full restart completion.
    try:
        time.sleep(0.1)
        probe = subprocess.run(
            ["systemctl", "--user", "show", service_name, "-p", "Id"],
            env=restart_env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout or "").strip()
            return f"Error: Restart scheduled but service probe failed: {detail}"
    except Exception as e:
        return f"Error: Restart scheduled but probe failed: {e}"

    return f"Upgrade scheduled. Service will restart shortly.\nBenefit: {benefit}"
