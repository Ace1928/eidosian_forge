#!/usr/bin/env python3
# 🌀 Eidosian Path Management
"""
Universal Path Management - One Source of Truth for Paths

This module provides universal path resolution capabilities
for all Doc Forge modules, ensuring consistent and accurate 
path handling across the entire system.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Union, Optional, List

# Self-aware logging
logger = logging.getLogger("doc_forge.paths")

# Global cache for performance
_REPO_ROOT: Optional[Path] = None
_DOCS_DIR: Optional[Path] = None

def get_repo_root() -> Path:
    """
    Get the repository root directory with unwavering precision.
    
    Returns:
        Path to repository root
    """
    global _REPO_ROOT
    
    if _REPO_ROOT is not None:
        return _REPO_ROOT
    
    # Starting points for search
    start_points = [
        Path.cwd(),                         # Current directory
        Path(__file__).resolve().parent,    # Module directory
        Path(sys.argv[0]).resolve().parent  # Script directory
    ]
    
    # Add any parent directories to check
    for point in list(start_points):  # Make copy to avoid modifying during iteration
        start_points.extend(point.parents[:4])  # Check a few parent levels
        
    # Remove duplicates while preserving order
    unique_starts = []
    for path in start_points:
        if path not in unique_starts:
            unique_starts.append(path)
            
    # Search for repo indicators from each starting point
    for directory in unique_starts:
        # Check if this is repo root by looking for key indicators
        is_repo_root = any((
            (directory / "docs").is_dir() and (directory / "src" / "doc_forge").is_dir(),
            (directory / ".git").is_dir() and (directory / "src").is_dir(),
            (directory / "setup.py").is_file() and (directory / "docs").is_dir()
        ))
        
        if is_repo_root:
            _REPO_ROOT = directory
            logger.debug(f"Found repo root at {_REPO_ROOT}")
            return _REPO_ROOT
    
    # Fallback: If we're in src/doc_forge, go up two levels
    current = Path(__file__).resolve().parent
    if "src/doc_forge" in str(current) or "src\\doc_forge" in str(current):
        candidate = current.parent.parent.parent
        if (candidate / "docs").is_dir():
            _REPO_ROOT = candidate
            logger.debug(f"Found repo root via module path: {_REPO_ROOT}")
            return _REPO_ROOT
    
    # Last resort: use current directory and warn
    _REPO_ROOT = Path.cwd()
    logger.warning(f"⚠️ Could not determine repo root, using current directory: {_REPO_ROOT}")
    return _REPO_ROOT

def get_docs_dir() -> Path:
    """
    Get the documentation directory with perfect precision.
    
    Returns:
        Path to documentation directory
    """
    global _DOCS_DIR
    
    if _DOCS_DIR is not None:
        return _DOCS_DIR
    
    # First, get the repo root
    repo_root = get_repo_root()
    
    # Check the canonical location first
    docs_dir = repo_root / "docs"
    if docs_dir.is_dir():
        _DOCS_DIR = docs_dir
        return _DOCS_DIR
        
    # Check fallback locations
    candidates = [
        repo_root / "documentation",
        repo_root / "doc"
    ]
    
    for candidate in candidates:
        if candidate.is_dir():
            _DOCS_DIR = candidate
            logger.warning(f"⚠️ Using non-standard docs directory: {_DOCS_DIR}")
            return _DOCS_DIR
    
    # Create it if it doesn't exist
    docs_dir.mkdir(parents=True, exist_ok=True)
    _DOCS_DIR = docs_dir
    logger.warning(f"⚠️ Created missing documentation directory: {_DOCS_DIR}")
    return _DOCS_DIR

def resolve_path(path: Union[str, Path], relative_to: Optional[Path] = None) -> Path:
    """
    Resolve a path with Eidosian precision.
    
    Args:
        path: Path to resolve
        relative_to: Base path for relative paths
        
    Returns:
        Resolved absolute path
    """
    path_obj = Path(path)
    
    # If already absolute, return as is
    if path_obj.is_absolute():
        return path_obj
    
    # If relative_to is specified, use it
    if relative_to is not None:
        return (relative_to / path_obj).resolve()
    
    # Otherwise, make relative to repo root
    return (get_repo_root() / path_obj).resolve()

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists with perfect precision.
    
    Args:
        path: Path to directory
        
    Returns:
        Absolute path to the directory
    """
    path_obj = resolve_path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def ensure_scripts_dir() -> Path:
    """
    Ensure the scripts directory exists with perfect precision.
    
    Returns:
        Path to the scripts directory
    """
    repo_root = get_repo_root()
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    return scripts_dir

# Initialize with debug info
logger.debug(f"Module {__name__} loaded")
logger.debug(f"Current working directory: {Path.cwd()}")
logger.debug(f"Module file location: {Path(__file__).resolve()}")
