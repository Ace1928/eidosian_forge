"""
Registry for Repo Forge templates and structures.
Integrates with GisCore for centralized configuration.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
from gis_forge import GisCore

class RepoRegistry:
    """
    Manages registration of repository templates and structures.
    """
    
    def __init__(self, gis: Optional[GisCore] = None):
        self.gis = gis or GisCore()
        self._setup_defaults()

    def _setup_defaults(self):
        """Register default Eidosian structures in GIS."""
        self.gis.set("repo.defaults.languages", ["python", "nodejs", "go", "rust"])
        self.gis.set("repo.defaults.core_dirs", [
            "projects", "libs", "tools", "scripts", "docs", "tests", 
            "benchmarks", "examples", "ci", ".github", "config", "shared"
        ])

    def register_template(self, name: str, content: str):
        """Register a content template in GIS."""
        self.gis.set(f"repo.templates.{name}", content)

    def get_template(self, name: str) -> Optional[str]:
        """Retrieve a template from GIS."""
        return self.gis.get(f"repo.templates.{name}")

    def register_structure(self, language: str, structure: Dict[str, Any]):
        """Register a language-specific directory structure."""
        self.gis.set(f"repo.structures.{language}", structure)

    def get_structure(self, language: str) -> Dict[str, Any]:
        """Get the structure for a language, falling back to defaults."""
        return self.gis.get(f"repo.structures.{language}", {})
