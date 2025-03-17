"""
Core utilities for Eidosian Repo Forge.

This package provides the foundational functionality for repository generation,
including file handling, directory creation, and template rendering.
"""
from .directory import create_directory_structure
from .files import write_file, generate_files
from .templates import render_template, render_comment_block, EidosianTemplate
from .utils import run_command, make_executable, detect_environment

__all__ = [
    # Directory operations
    "create_directory_structure",
    
    # File operations
    "write_file",
    "generate_files",
    
    # Template handling
    "render_template",
    "render_comment_block",
    "EidosianTemplate",
    
    # Utilities
    "run_command",
    "make_executable",
    "detect_environment",
]
