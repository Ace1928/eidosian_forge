"""
Utility modules for Figlet Forge providing reusable functionality.

This package contains various utility modules that support the
core functionality of Figlet Forge, following Eidosian principles
of modularity, precision, and elegance.
"""

from .file_utils import (
    get_default_font_dir,
    get_font_path,
    list_font_files,
    resolve_resource_path,
)

__all__ = [
    "get_default_font_dir",
    "list_font_files",
    "get_font_path",
    "resolve_resource_path",
]
