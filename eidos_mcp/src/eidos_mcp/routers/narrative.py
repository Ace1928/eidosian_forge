from __future__ import annotations

from typing import Optional
from eidosian_core import eidosian
from ..core import tool
from ..state import FORGE_DIR

try:
    from narrative_forge.engine import NarrativeEngine
except ImportError:
    import sys
    sys.path.append(str(FORGE_DIR / "narrative_forge/src"))
    from narrative_forge.engine import NarrativeEngine

# Singleton engine instance - defaults to local ollama for safety/privacy
_engine = NarrativeEngine(provider="ollama", model_name="qwen3.5:2b")

@eidosian()
@tool(name="narrative_respond", description="Interact with the Eidosian Narrative Engine.")
def narrative_respond(user_input: str) -> str:
    """Get a response from the self-referential narrative engine."""
    return _engine.respond(user_input)

@eidosian()
@tool(name="narrative_free_thought", description="Trigger an autonomous reflection from the narrative engine.")
def narrative_free_thought() -> str:
    """Generate an autonomous thought based on recent interactions."""
    _engine.free_thought()
    if _engine.store.data.events:
        return _engine.store.data.events[-1]["event"]
    return "The void is silent."

@eidosian()
@tool(name="narrative_status", description="Get the current status of the narrative memory.")
def narrative_status() -> dict:
    """Return metrics on interactions and stored events."""
    return {
        "interactions_count": len(_engine.store.data.interactions),
        "events_count": len(_engine.store.data.events),
        "model": _engine.model_name,
        "provider": _engine.provider_name
    }
