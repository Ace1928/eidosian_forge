from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from eidosian_core import eidosian

from .. import FORGE_ROOT
from ..core import tool
from ..state import agent


@tool(
    name="agent_run_task",
    description="Delegate a complex objective to the Agent Forge. Returns a plan or execution summary.",
    parameters={
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "execute": {"type": "boolean", "description": "If true, execute the plan immediately."},
        },
        "required": ["objective"],
    },
)
@eidosian()
def agent_run_task(objective: str, execute: bool = False) -> str:
    """Delegate a complex objective to the Agent Forge."""
    if not agent:
        return json.dumps(
            {
                "objective": objective,
                "status": "unavailable",
                "execute": execute,
                "tasks": [],
                "error": "Agent Forge not available (import failed)",
            },
            indent=2,
        )

    goal = agent.create_goal(objective, plan=True)

    execution_results = []
    if execute:
        for t in goal.tasks:
            success = agent.execute_task(t)
            execution_results.append(
                {
                    "task": t.description,
                    "tool": t.tool,
                    "status": t.status,
                    "result": str(t.result)[:200] + "..." if t.result and len(str(t.result)) > 200 else t.result,
                }
            )
            if not success and t.priority > 0:  # Stop on critical failure
                break

        return json.dumps({"objective": objective, "status": "executed", "results": execution_results}, indent=2)

    # We return the plan for inspection.
    # In a full autonomous loop, we might execute it, but for MCP we prefer planning first.
    tasks = []
    for t in goal.tasks:
        tasks.append({"description": t.description, "tool": t.tool, "args": t.kwargs, "status": t.status})

    return json.dumps({"objective": objective, "plan_id": str(id(goal)), "tasks": tasks}, indent=2)


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
                timeout=300,  # 5 minutes max for tests
            )
            if result.returncode != 0:
                return f"Error: Tests failed. Upgrade aborted.\n\nStdout:\n{result.stdout}\n\nStderr:\n{result.stderr}"
        except Exception as e:
            return f"Error: Failed to run tests: {e}"

    # If we got here, tests passed or were skipped. Restart the service.
    # Check if systemctl is available
    has_systemctl = subprocess.run(["command", "-v", "systemctl"], shell=True, capture_output=True).returncode == 0

    if has_systemctl:
        # Use explicit user bus in case the MCP process runs without it in env.
        service_name = os.environ.get("EIDOS_MCP_SYSTEMD_SERVICE", "eidos-mcp.service")
        user_bus = os.environ.get("DBUS_SESSION_BUS_ADDRESS") or f"unix:path=/run/user/{os.getuid()}/bus"
        restart_env = dict(os.environ)
        restart_env["DBUS_SESSION_BUS_ADDRESS"] = user_bus

        # Return success before the process potentially restarts itself.
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
    else:
        # Termux/Direct Process Fallback
        try:
            server_script = Path(forge_dir) / "eidos_mcp" / "run_server.sh"
            # We use pkill -f to find the server process and then launch the script in background
            subprocess.Popen(
                [
                    "bash",
                    "-c",
                    f"sleep 1 && pkill -f eidos_mcp.eidos_mcp_server ; {server_script} &",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            return f"Error: Failed to schedule direct restart: {e}"

    return f"Upgrade scheduled. Service will restart shortly.\nBenefit: {benefit}"


@tool(
    name="nexus_map_forge",
    description="Formally map a forge module into the Unified Eidosian Ontology and index its components.",
    parameters={
        "type": "object",
        "properties": {
            "forge_name": {"type": "string", "description": "The directory name of the forge (e.g., 'figlet_forge')"},
            "functional_cluster": {
                "type": "string",
                "enum": ["Cognitive Spine", "Perceptual/Actuator", "Communication", "Infrastructure"],
                "description": "The architectural cluster this forge belongs to"
            },
            "status": {"type": "string", "enum": ["Active", "Legacy", "Metabolized"], "description": "Current operational status"},
            "summary": {"type": "string", "description": "One-sentence technical summary of the forge's purpose"}
        },
        "required": ["forge_name", "functional_cluster", "status", "summary"]
    }
)
@eidosian()
def nexus_map_forge(
    forge_name: str,
    functional_cluster: str,
    status: str,
    summary: str
) -> str:
    """Metabolic mapping of a forge into the systemic ontology."""
    from .knowledge import kb_add_fact
    from .code import code_index_directory
    
    # 1. Ontological Linkage
    fact = f"FORGE_COMPONENT: {forge_name} | CLUSTER: {functional_cluster} | STATUS: {status} | SUMMARY: {summary}"
    kb_res = kb_add_fact(fact, tags=["ontology", "forge_map", forge_name])
    
    # 2. Code Indexing (if not metabolized)
    code_res = ""
    if status != "Metabolized":
        try:
            code_res = code_index_directory(f"./{forge_name}")
        except Exception as e:
            code_res = f"Code indexing failed: {e}"
            
    return json.dumps({
        "forge": forge_name,
        "ontology_result": kb_res,
        "indexing_result": code_res,
        "message": f"Forge {forge_name} has been formally mapped as a {functional_cluster} organ."
    }, indent=2)

