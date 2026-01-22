from __future__ import annotations

import os
import subprocess
import json
from typing import Optional

from ..core import tool
from ..state import agent


@tool(
    name="agent_run_task",
    description="Delegate a complex objective to the Agent Forge. Returns a plan or execution summary.",
    parameters={
        "type": "object",
        "properties": {"objective": {"type": "string"}},
        "required": ["objective"],
    },
)
def agent_run_task(objective: str) -> str:
    """Delegate a complex objective to the Agent Forge."""
    if not agent:
        return json.dumps({"error": "Agent Forge not available (import failed)"})
    
    goal = agent.create_goal(objective, plan=True)
    
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

    forge_dir = os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge")
    
    if run_tests:
        # Run tests in the forge directory
        try:
            # We use the venv python to run pytest
            venv_python = "/home/lloyd/eidosian_venv/bin/python3"
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
    try:
        # We use systemctl --user
        subprocess.run(
            ["systemctl", "--user", "restart", "eidos-mcp"],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        return f"Error: Failed to restart service: {e.stderr}"

    return f"Upgrade successful. Service restarted.\nBenefit: {benefit}"
