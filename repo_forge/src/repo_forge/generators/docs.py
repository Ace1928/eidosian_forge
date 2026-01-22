"""
Documentation structure generator for Eidosian repositories.

This module creates a comprehensive documentation structure following
the universal Eidosian documentation principles.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ..core.directory import create_directory
from ..core.files import write_file
from ..constants.paths import (
    DOCUMENTATION_STRUCTURE,
    MANUAL_DOC_STRUCTURE,
    AUTO_DOC_STRUCTURE,
    ASSETS_STRUCTURE
)


def create_documentation_structure(
    base_path: Path,
    languages: Optional[List[str]] = None,
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Create a comprehensive documentation structure.
    
    Args:
        base_path: Base repository directory
        languages: Programming languages to support documentation for
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with creation results
    """
    if languages is None:
        languages = ["python", "cpp", "rust", "go", "javascript"]
    
    docs_path = base_path / "docs"
    docs_path.mkdir(exist_ok=True, parents=True)
    
    created_dirs = []
    
    # Create common documentation structure
    for structure in DOCUMENTATION_STRUCTURE:
        path = docs_path / structure
        path.mkdir(exist_ok=True, parents=True)
        created_dirs.append(str(path.relative_to(base_path)))
    
    # Create language-specific documentation
    for language in languages:
        # Manual documentation
        manual_lang_path = docs_path / "manual" / language
        manual_lang_path.mkdir(exist_ok=True, parents=True)
        
        for subdir in MANUAL_DOC_STRUCTURE:
            subdir_path = manual_lang_path / subdir
            subdir_path.mkdir(exist_ok=True, parents=True)
            created_dirs.append(str(subdir_path.relative_to(base_path)))
        
        # Create index.md file
        index_content = f"""# {language.title()} Documentation

Welcome to the {language.title()} documentation for this project.

## Contents

- [Guides](guides/): Step-by-step tutorials
- [API Documentation](api/): Detailed API reference
- [Design](design/): Architecture and design documents
- [Examples](examples/): Code examples
- [Best Practices](best_practices/): Recommended patterns and practices
- [Troubleshooting](troubleshooting/): Common issues and solutions
- [Security](security/): Security guidelines and considerations
- [Changelog](changelog/): Version history
- [Contributing](contributing/): How to contribute
- [FAQ](faq/): Frequently Asked Questions
"""
        write_file(manual_lang_path / "index.md", index_content, overwrite)
        
        # Auto-generated documentation
        auto_lang_path = docs_path / "auto" / language
        auto_lang_path.mkdir(exist_ok=True, parents=True)
        
        for subdir in AUTO_DOC_STRUCTURE:
            subdir_path = auto_lang_path / subdir
            subdir_path.mkdir(exist_ok=True, parents=True)
            created_dirs.append(str(subdir_path.relative_to(base_path)))
        
        # Create auto index.md file
        auto_index_content = f"""# Auto-Generated {language.title()} Documentation

This section contains automatically generated documentation for the {language.title()} code.

## Contents

- [API Reference](api/): Auto-generated API documentation
- [Data Models](models/): Documentation for data models
- [Functions](functions/): Function-level documentation
- [Error Handling](error_handling/): Exception and error documentation
- [Benchmarks](benchmarks/): Performance benchmarks
- [Internal API](internal/): Documentation for internal APIs
- [Schemas](schemas/): Database and data structure schemas
- [Configuration](configuration/): Configuration options and reference
"""
        write_file(auto_lang_path / "index.md", auto_index_content, overwrite)
    
    # Create assets structure
    assets_path = docs_path / "assets"
    for subdir in ASSETS_STRUCTURE:
        create_directory(assets_path / subdir)
        created_dirs.append(f"docs/assets/{subdir}")
    
    # Create README for assets
    assets_readme = """# Documentation Assets

This directory contains static assets used in the documentation:

- `images/`: Screenshots, illustrations, and other images
- `diagrams/`: Architecture diagrams, flowcharts, and UML diagrams
- `css/`: Custom stylesheets for documentation
- `fonts/`: Custom fonts used in documentation
"""
    write_file(assets_path / "README.md", assets_readme)
    
    # Create main documentation index
    main_index_content = """# Project Documentation

Welcome to the comprehensive documentation for this project.

## Structure

- [Manual Documentation](manual/): Hand-written documentation
- [Auto-Generated Documentation](auto/): Documentation generated from code
- [Assets](assets/): Images, diagrams, and other static assets

## Languages

"""
    for language in languages:
        main_index_content += f"- [{language.title()}](manual/{language}/): {language.title()} documentation\n"
    
    write_file(docs_path / "index.md", main_index_content)
    
    # Create Sphinx configuration
    sphinx_conf = """# Configuration file for the Sphinx documentation builder

project = "Eidosian Project"
copyright = "2025, Eidosian"
author = "Eidosian"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
"""
    write_file(docs_path / "conf.py", sphinx_conf)
    
    # Create ReadTheDocs config
    rtd_config = """version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.10"

sphinx:
  configuration: docs/conf.py

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
"""
    write_file(docs_path / ".readthedocs.yaml", rtd_config)
    
    # Create Sphinx-required directories for full compatibility
    sphinx_dirs = ["_static", "_templates"]
    for sphinx_dir in sphinx_dirs:
        sphinx_path = docs_path / sphinx_dir
        sphinx_path.mkdir(exist_ok=True, parents=True)
        created_dirs.append(str(sphinx_path.relative_to(base_path)))
    
    # Add source directory structure for compatibility with universal standard
    source_path = docs_path / "source"
    source_path.mkdir(exist_ok=True, parents=True)
    
    for section in ["concepts", "examples", "getting_started", "guides", "reference", "architecture"]:
        section_path = source_path / section
        section_path.mkdir(exist_ok=True, parents=True)
        created_dirs.append(str(section_path.relative_to(base_path)))

    # Create source/index.rst
    source_index = """# Source Documentation

This directory contains documentation source files organized by topic:

- [Concepts](concepts/): Core concepts and principles
- [Examples](examples/): Example code and tutorials
- [Getting Started](getting_started/): Quickstart guides
- [Guides](guides/): In-depth guides
- [Reference](reference/): API reference documentation
- [Architecture](architecture/): Architectural overviews
"""
    write_file(source_path / "index.md", source_index, overwrite)
    
    logging.info(f"Created {len(created_dirs)} documentation directories")
    return {
        "success": True,
        "created_dirs": created_dirs,
        "languages": languages,
        "base_path": str(base_path)
    }