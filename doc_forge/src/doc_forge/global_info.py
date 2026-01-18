#!/usr/bin/env python3
# 🌀 Eidosian Global Information System
"""
Global Information System for Doc Forge

This module provides global configuration and project-wide information
for the Doc Forge system. It serves as a central point of truth for
configuration, paths, and standards across the entire system.

Following Eidosian principles of:
- Contextual Integrity: Every piece of information has precise purpose
- Structure as Control: Perfect organization of project architecture
- Self-Awareness as Foundation: System that knows its own configuration
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any

# Import version information for consistency
from .version import VERSION, get_version_string

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 Self-aware logging system
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logger = logging.getLogger("doc_forge.global_info")
logger.info(f"Doc Forge version initialized: {get_version_string()}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🏗️ Project structure configuration - The architecture of our universe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Project information
PROJECT = {
    "name": "Doc Forge",
    "description": "Universal Documentation Management System with Eidosian principles",
    "version": VERSION,
    "author": "Lloyd Handyside",
    "email": "ace1928@gmail.com",
    "organization": "Neuroforge",
    "org_email": "lloyd.handyside@neuroforge.io",
    "url": "https://doc-forge.readthedocs.io/",
    "license": "MIT",
    "copyright": f"2024, MIT License",
}

# Documentation directory structure
DOC_STRUCTURE = {
    "root": "docs",
    "build": "_build",
    "static": "_static",
    "templates": "_templates",
    "user_docs": "user_docs",
    "auto_docs": "auto_docs",
    "ai_docs": "ai_docs",
    "assets": "assets",
}

# Document categories with sections
DOC_CATEGORIES = {
    "user_docs": [
        "getting_started",
        "guides",
        "concepts",
        "reference",
        "examples",
        "faq"
    ],
    "auto_docs": [
        "api",
        "introspected",
        "extracted"
    ],
    "ai_docs": [
        "generated",
        "enhanced",
        "integrated"
    ]
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔧 Configuration settings - The rules of our universe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_CONFIG: Dict[str, Any] = {
    "theme": "sphinx_rtd_theme",
    "extensions": [
        "sphinx.ext.autodoc",
        "sphinx.ext.viewcode",
        "sphinx.ext.napoleon",
        "sphinx.ext.autosummary",
        "sphinx.ext.intersphinx",
        "sphinx.ext.coverage",
        "sphinx.ext.todo",
        "sphinx.ext.ifconfig",
        "myst_parser",
        "sphinx_rtd_theme",
        "sphinx_copybutton",
        "sphinx_autodoc_typehints",
        "sphinxcontrib.mermaid",
        "sphinx_autoapi",
        "sphinx.ext.autosectionlabel",
    ],
    "formats": ["html", "pdf", "epub"],
    "language": "en",
    "master_doc": "index",
    "autoapi_dirs": ["src"],
    "autoapi_template_dir": "auto_docs",
}  # type: ignore

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧪 Functions for working with global information
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_project_info() -> Dict[str, str]:
    """
    Get project information with Eidosian clarity.
    
    Returns:
        Dictionary with project information
    """
    return PROJECT

def get_doc_structure(repo_root: Optional[Path] = None) -> Dict[str, Path]:
    """
    Get documentation directory structure with absolute paths.
    
    Args:
        repo_root: Repository root path (auto-detected if None)
        
    Returns:
        Dictionary mapping structure keys to absolute paths
    """
    if repo_root is None:
        from .utils.paths import get_repo_root
        repo_root = get_repo_root()
    
    result: Dict[str, Path] = {}
    
    # Add the root docs directory
    docs_dir = repo_root / DOC_STRUCTURE["root"]
    result["root"] = docs_dir
    
    # Add all subdirectories
    for key, subdir in DOC_STRUCTURE.items():
        if key != "root":
            result[key] = docs_dir / subdir
    
    # Add all category subdirectories
    for category, sections in DOC_CATEGORIES.items():
        category_dir = docs_dir / DOC_STRUCTURE.get(category, category)
        result[category] = category_dir
        
        # Add sections within each category
        for section in sections:
            result[f"{category}_{section}"] = category_dir / section
    
    return result

def get_config() -> Dict[str, Any]:
    """
    Get configuration with environment overrides.
    
    Returns:
        Dictionary with configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables
    for key, value in config.items():
        env_key = f"DOC_FORGE_{key.upper()}"
        if env_key in os.environ:
            env_value = os.environ[env_key]
            
            # Convert to appropriate type
            if isinstance(value, list):
                config[key] = env_value.split(",")
            elif isinstance(value, bool):
                config[key] = env_value.lower() in ("true", "1", "yes", "y")
            elif isinstance(value, int):
                config[key] = int(env_value)
            else:
                config[key] = env_value
    
    return config

def ensure_doc_structure(repo_root: Optional[Path] = None) -> Dict[str, Path]:
    """
    Ensure all documentation directories exist with Eidosian perfection.
    
    Args:
        repo_root: Repository root path (auto-detected if None)
        
    Returns:
        Dictionary mapping structure keys to absolute paths
    """
    structure = get_doc_structure(repo_root)
    
    # Create all directories
    for path in structure.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return structure

# Initialize global paths dictionary on module load
_global_paths: Dict[str, Any] = {}
GLOBAL_PATHS_OVERRIDE: Dict[str, Any] = {}

def get_paths() -> Dict[str, Path]:
    """
    Get global paths with Eidosian precision.
    
    Returns:
        Dictionary mapping path keys to absolute paths
    """
    global _global_paths
    
    if not _global_paths:
        try:
            from .utils.paths import get_repo_root
            repo_root = get_repo_root()
            _global_paths = get_doc_structure(repo_root)
        except Exception as e:
            logger.error(f"❌ Failed to initialize global paths: {e}")
            _global_paths = {}
    
    return _global_paths

def get_global_config() -> Dict[str, Any]:
    """
    Get global configuration with Eidosian clarity.
    
    Returns:
        Dictionary with global configuration
    """
    return {}  # type: ignore  # Minimizing changes; ignoring partial unknown type
