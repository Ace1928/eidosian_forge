from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from eidosian_core import eidosian
from ..core import tool
from ..state import FORGE_DIR

try:
    from repo_forge.generators import project, docs
except ImportError:
    import sys
    sys.path.append(str(FORGE_DIR / "repo_forge/src"))
    from repo_forge.generators import project, docs

@eidosian()
@tool(name="repo_create_scaffold", description="Generate a project scaffold for a specific language.")
def repo_create_scaffold(language: str, base_path: Optional[str] = None, overwrite: bool = True) -> Dict[str, Any]:
    """
    Creates a standard project structure (src, tests, README, etc.).
    
    Args:
        language: 'python', 'nodejs', 'go', 'rust'.
        base_path: The root directory for the project (defaults to workspace root).
        overwrite: Whether to overwrite existing files.
    """
    root = Path(base_path) if base_path else FORGE_DIR.parent
    return project.create_project_scaffold_single(base_path=root, language=language, overwrite=overwrite)

@eidosian()
@tool(name="repo_create_docs", description="Generate a comprehensive documentation structure.")
def repo_create_docs(base_path: Optional[str] = None, languages: Optional[List[str]] = None, overwrite: bool = True) -> Dict[str, Any]:
    """
    Creates a full Eidosian documentation hierarchy (manual, auto, assets, etc.).
    
    Args:
        base_path: Root directory for documentation (defaults to workspace root).
        languages: List of languages to support (e.g., ['python', 'cpp']).
        overwrite: Whether to overwrite existing index files.
    """
    root = Path(base_path) if base_path else FORGE_DIR.parent
    return docs.create_documentation_structure(base_path=root, languages=languages, overwrite=overwrite)
