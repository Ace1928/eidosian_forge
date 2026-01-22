"""
🔮 Eidosian Repo Forge

A meticulously engineered Python toolkit for creating universal monorepo structures
that fully embody all Eidosian principles.

This package transforms repository creation from ad-hoc scripting to a precise,
systematic architecture that enforces best practices while remaining infinitely adaptable.
"""

__version__ = "0.1.0"

# Core utilities
from .core.directory import create_directory_structure
from .core.files import generate_files, write_file
from .core.templates import render_template, render_comment_block
from .core.utils import run_command, make_executable, detect_environment

# Generators
from .generators.config import create_configuration_files, create_eidosian_json
from .generators.docs import create_documentation_structure
from .generators.project import create_project_scaffold
from .generators.scripts import create_script_files

__all__ = [
    # Core utilities
    "create_directory_structure",
    "generate_files",
    "write_file",
    "render_template",
    "render_comment_block",
    "run_command",
    "make_executable",
    "detect_environment",
    
    # Generators
    "create_configuration_files",
    "create_eidosian_json", 
    "create_documentation_structure", 
    "create_project_scaffold",
    "create_script_files"
]