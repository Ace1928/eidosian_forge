"""
Eidosian Refactor - Intelligent Code Transformation Engine 🧠

Transforms monolithic code into perfectly structured modular architecture
with mathematical precision and Eidosian principles.

Core capabilities:
- AST-based code analysis for natural boundary detection
- Dependency mapping for perfect relationship preservation 
- Contextual documentation generation with Eidosian awareness
- Lossless transformation with full functional equivalence

Embodies Eidosian principles:
- Contextual Integrity: Every component knows its place in the ecosystem
- Precision as Style: Structural perfection with zero compromise
- Flow Like a River: Natural code organization for frictionless development
- Structure as Control: Architectural harmony that amplifies human capability

Created by: Lloyd Handyside & Eidos
Last update: 2025-03-12
"""

# <!-- VERSION_START -->
# Version is managed through pyproject.toml - dynamically loaded here
try:
    from importlib.metadata import version as _version
    __version__ = _version("eidosian-refactor")
except ImportError:  # pragma: no cover
    # Fallback for Python < 3.8
    try:
        import pkg_resources
        __version__ = pkg_resources.get_distribution("eidosian-refactor").version
    except Exception:  # pragma: no cover
        # If all else fails, provide a basic version
        __version__ = "0.1.0"  # FALLBACK_VERSION - Updated by version_update.py
# <!-- VERSION_END -->
