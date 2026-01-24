from __future__ import annotations

import json
import os
from typing import Dict, Any

from ..core import tool
from eidosian_core import eidosian


@tool(
    name="auth_whoami",
    description="Returns the current authentication status and identity.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
@eidosian()
def auth_whoami() -> str:
    """
    Returns the current authentication status.
    
    In a real implementation with Context injection, this would inspect 
    request headers for 'Authorization: Bearer ...'.
    
    For now, it reports the transport and potential identity.
    """
    transport = os.environ.get("EIDOS_MCP_TRANSPORT", "unknown")
    
    # Placeholder for future context injection
    identity = "anonymous"
    
    # Check if we are running in an environment where we expect auth
    auth_mode = "open"
    if transport == "sse":
        auth_mode = "bearer_token_ready"
        
    return json.dumps({
        "transport": transport,
        "mode": auth_mode,
        "identity": identity,
        "note": "To enable strict Google Auth, configure 'authProviderType: google_credentials' in Gemini settings."
    }, indent=2)
