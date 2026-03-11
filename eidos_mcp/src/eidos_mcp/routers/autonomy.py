from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from eidosian_core import eidosian

from .. import FORGE_ROOT
from ..core import tool
from ..forge_loader import ensure_forge_import

ensure_forge_import("agent_forge")

try:
    from agent_forge.autonomy.gates import SystemicGateKeeper
except ImportError:
    SystemicGateKeeper = None

FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", str(FORGE_ROOT))).resolve()

# Initialize GateKeeper
_gatekeeper: Optional[SystemicGateKeeper] = None

def _get_gatekeeper() -> Optional[SystemicGateKeeper]:
    global _gatekeeper
    if _gatekeeper is None and SystemicGateKeeper is not None:
        _gatekeeper = SystemicGateKeeper(
            repo_root=FORGE_DIR,
            invariants_path=FORGE_DIR / "GEMINI.md"
        )
    return _gatekeeper

@tool(
    name="autonomy_propose_modification",
    description="Propose a systemic modification (code, config, or identity) for formal gating.",
    parameters={
        "type": "object",
        "properties": {
            "target_path": {"type": "string", "description": "Path to the file or component being modified"},
            "change_type": {
                "type": "string", 
                "enum": ["code", "config", "identity"],
                "description": "The category of systemic change"
            },
            "proposed_content": {"type": "string", "description": "The new content or delta being proposed"},
            "rationale": {"type": "string", "description": "Technical and strategic justification for the change"}
        },
        "required": ["target_path", "change_type", "proposed_content", "rationale"]
    }
)
@eidosian()
def propose_modification(
    target_path: str,
    change_type: str,
    proposed_content: str,
    rationale: str
) -> str:
    """Formal proposal for systemic modification."""
    gk = _get_gatekeeper()
    if not gk:
        return "Error: Systemic GateKeeper is unavailable."
    
    proposal_id = gk.propose_change(
        target_path=target_path,
        change_type=change_type,
        proposed_content=proposed_content,
        rationale=rationale
    )
    return f"Proposal {proposal_id} registered. Call autonomy_validate_proposal to continue."

@tool(
    name="autonomy_validate_proposal",
    description="Execute the formal validation battery for a registered modification proposal.",
    parameters={
        "type": "object",
        "properties": {
            "proposal_id": {"type": "string", "description": "The ID of the proposal to validate"}
        },
        "required": ["proposal_id"]
    }
)
@eidosian()
def validate_proposal(proposal_id: str) -> str:
    """Validate a modification proposal."""
    gk = _get_gatekeeper()
    if not gk:
        return "Error: Systemic GateKeeper is unavailable."
    
    import json
    result = gk.validate_proposal(proposal_id)
    return json.dumps(result, indent=2)
