"""
Generators for Eidosian Repo Forge.

This package contains modules that generate different aspects of an
Eidosian repository structure, from documentation to configuration files.
"""
from .config import create_configuration_files, create_eidosian_json
from .docs import create_documentation_structure
from .project import create_project_scaffold
from .scripts import create_script_files

__all__ = [
    "create_configuration_files",
    "create_eidosian_json",
    "create_documentation_structure",
    "create_project_scaffold",
    "create_script_files",
]
