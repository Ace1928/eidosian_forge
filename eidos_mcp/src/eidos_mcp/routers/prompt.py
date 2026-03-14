from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from eidosian_core import eidosian
from ..core import tool
from ..state import FORGE_DIR

try:
    from prompt_forge.library import PromptLibrary
except ImportError:
    # Handle the case where prompt_forge isn't in the path yet.
    # We use a lazy import/init approach.
    import sys
    sys.path.append(str(FORGE_DIR / "prompt_forge/src"))
    from prompt_forge.library import PromptLibrary

# Singleton library instance
_library = PromptLibrary()

@eidosian()
@tool(name="prompt_get", description="Retrieve a prompt template by name.")
def prompt_get(name: str) -> Optional[str]:
    """Retrieve a prompt template by name."""
    return _library.get_prompt(name)

@eidosian()
@tool(name="prompt_list", description="List all available prompt templates.")
def prompt_list() -> List[str]:
    """List available prompts."""
    return _library.list_prompts()

@eidosian()
@tool(name="prompt_save", description="Save or update a prompt template.")
def prompt_save(name: str, content: str) -> str:
    """Save a new prompt template."""
    _library.save_prompt(name, content)
    return f"Successfully saved prompt: {name}"
