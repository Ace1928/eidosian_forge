"""
FigletString interface for backward compatibility.

This module re-exports the FigletString class from the core module for
backward compatibility with code that imports directly from figlet_forge.
"""

from .core.figlet_string import FigletString

__all__ = ["FigletString"]
