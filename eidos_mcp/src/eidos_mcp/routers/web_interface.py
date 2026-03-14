from __future__ import annotations

from typing import Dict, Any
from eidosian_core import eidosian
from ..core import tool
from ..state import FORGE_DIR

@eidosian()
@tool(name="web_interface_status", description="Get the status of the Eidos Hybrid Chat Sidecar.")
def web_interface_status() -> Dict[str, Any]:
    """
    Returns information about the web interface sidecar server.
    Note: The actual Playwright server must be running independently.
    """
    import os
    state_path = os.path.expanduser("~/.eidos_chatgpt_state.json")
    has_state = os.path.exists(state_path)
    
    return {
        "architecture": "Playwright Headed Browser + WebSocket Bridge",
        "default_port": 8932,
        "state_file_exists": has_state,
        "state_path": state_path,
        "instructions": "Run 'python eidos_server.py' in web_interface_forge to start the sidecar."
    }
