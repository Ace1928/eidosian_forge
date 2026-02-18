from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, List
from eidosian_core import eidosian

class PromptLibrary:
    """Manages Eidosian prompt templates."""

    def __init__(self, library_path: Optional[Path] = None):
        self.library_path = library_path or Path(__file__).resolve().parents[4] / "data/prompts"
        self.library_path.mkdir(parents=True, exist_ok=True)

    @eidosian()
    def get_prompt(self, name: str) -> Optional[str]:
        """Retrieve a prompt template by name."""
        target = self.library_path / f"{name}.txt"
        if not target.exists():
            # Fallback to check for .md or no extension
            target_md = self.library_path / f"{name}.md"
            if target_md.exists():
                target = target_md
            else:
                target_base = self.library_path / name
                if target_base.exists():
                    target = target_base
                else:
                    return None
        
        return target.read_text(encoding="utf-8")

    @eidosian()
    def list_prompts(self) -> List[str]:
        """List available prompts."""
        if not self.library_path.exists():
            return []
        return [p.stem for p in self.library_path.glob("*.*") if p.is_file()]

    @eidosian()
    def save_prompt(self, name: str, content: str) -> None:
        """Save a new prompt template."""
        target = self.library_path / f"{name}.txt"
        target.write_text(content, encoding="utf-8")
