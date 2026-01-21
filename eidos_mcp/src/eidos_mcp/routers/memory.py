from typing import Optional, List
from ..core import mcp
from memory_forge import MemoryForge, MemoryConfig
from pathlib import Path
import os

# Initialize Forge
# In a real app this might be dependency injected, but for now we instantiate here
# assuming standard paths or env vars.
FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge"))
memory = MemoryForge(config=MemoryConfig(
    episodic={"connection_string": str(FORGE_DIR / "memory_data.json"), "type": "json"}
))

@mcp.tool()
def memory_add(content: str, is_fact: bool = False, key: Optional[str] = None) -> str:
    """Add a new memory (episodic or semantic)."""
    # Simply using the episodic backend for now
    mid = memory.remember(content, embedding=[0.0]*10) # Placeholder embedding until LLM integration
    return f"Memory added with ID: {mid}"

@mcp.tool()
def memory_retrieve(query: str) -> str:
    """Search for relevant memories by query."""
    # Placeholder embedding
    results = memory.recall([0.0]*10)
    return str([r.to_dict() for r in results])
