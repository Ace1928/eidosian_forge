#!/usr/bin/env python3
# 🌀 Eidosian Version System - Single Source of Truth
"""
Version information for Doc Forge.

This module provides a single source of truth for version information
across the entire Doc Forge system. It follows Eidosian principles of
precision, structure, and universal applicability.
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple, Union, Optional

# Version components with Eidosian precision
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0
VERSION_LABEL = "alpha"  # Can be "alpha", "beta", "rc", etc.
VERSION_LABEL_NUM = 0  # For alpha.1, beta.2, etc.

# Assembled version information - the core truth
__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
if VERSION_LABEL:
    __version__ += f"-{VERSION_LABEL}"
    if VERSION_LABEL_NUM > 0:
        __version__ += f".{VERSION_LABEL_NUM}"

# PEP 440 compatible version for setuptools
__pep440_version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
if VERSION_LABEL:
    if VERSION_LABEL == "alpha":
        __pep440_version__ += f"a{VERSION_LABEL_NUM}" if VERSION_LABEL_NUM > 0 else "a0"
    elif VERSION_LABEL == "beta":
        __pep440_version__ += f"b{VERSION_LABEL_NUM}" if VERSION_LABEL_NUM > 0 else "b0"
    elif VERSION_LABEL == "rc":
        __pep440_version__ += f"rc{VERSION_LABEL_NUM}" if VERSION_LABEL_NUM > 0 else "rc0"
    else:
        __pep440_version__ += f".{VERSION_LABEL}{VERSION_LABEL_NUM}" if VERSION_LABEL_NUM > 0 else f".{VERSION_LABEL}"

# Global version variables for consistency
VERSION = __version__
PEP440_VERSION = __pep440_version__

def get_version_string() -> str:
    """
    Get the full version string with Eidosian clarity.
    
    Returns:
        Complete version string
    """
    return VERSION

def get_version_tuple() -> Tuple[int, int, int, str, int]:
    """
    Get the version components as a tuple.
    
    Returns:
        Tuple of (major, minor, patch, label, label_number)
    """
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_LABEL, VERSION_LABEL_NUM)

def get_version_info() -> Dict[str, Union[int, str]]:
    """
    Get complete version information as a dictionary.
    
    Returns:
        Dictionary with version components
    """
    return {
        "major": VERSION_MAJOR,
        "minor": VERSION_MINOR,
        "patch": VERSION_PATCH,
        "label": VERSION_LABEL,
        "label_num": VERSION_LABEL_NUM,
        "version": VERSION,
        "pep440_version": PEP440_VERSION
    }

def get_version_from_file() -> Optional[str]:
    """
    Get version from VERSION file if it exists.
    
    Returns:
        Version string from file or None if file doesn't exist
    """
    # Try to find VERSION file in common locations
    version_file_paths = [
        Path(__file__).resolve().parent.parent.parent / "VERSION",  # /repo/VERSION
        Path(__file__).resolve().parent.parent / "VERSION",         # /repo/src/VERSION
        Path(__file__).resolve().parent / "VERSION",                # /repo/src/doc_forge/VERSION
    ]
    
    for version_file in version_file_paths:
        if version_file.exists():
            with open(version_file, "r", encoding="utf-8") as f:
                return f.read().strip()
    
    return None

def update_version_from_env() -> None:
    """Update global version variables from environment variables."""
    global VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_LABEL, VERSION_LABEL_NUM, VERSION, PEP440_VERSION
    
    if "DOC_FORGE_VERSION" in os.environ:
        version_str = os.environ["DOC_FORGE_VERSION"]
        
        # Parse version string with regex
        match = re.match(r"(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z]+)\.?(\d+)?)?", version_str)
        if match:
            groups = match.groups()
            __version_major__ = int(groups[0])
            __version_minor__ = int(groups[1])
            __version_patch__ = int(groups[2])
            __version_label__ = groups[3] or ""
            __version_label_num__ = int(groups[4]) if groups[4] else 0
            
            # Update assembled versions
            __version__ = f"{__version_major__}.{__version_minor__}.{__version_patch__}"
            if __version_label__:
                __version__ += f"-{__version_label__}"
                if __version_label_num__ > 0:
                    __version__ += f".{__version_label_num__}"
            
            # Update PEP440 version
            __pep440_version__ = f"{__version_major__}.{__version_minor__}.{__version_patch__}"
            if __version_label__:
                if __version_label__ == "alpha":
                    __pep440_version__ += f"a{__version_label_num__}" if __version_label_num__ > 0 else "a0"
                elif __version_label__ == "beta":
                    __pep440_version__ += f"b{__version_label_num__}" if __version_label_num__ > 0 else "b0"
                elif __version_label__ == "rc":
                    __pep440_version__ += f"rc{__version_label_num__}" if __version_label_num__ > 0 else "rc0"
                else:
                    __pep440_version__ += f".{__version_label__}{__version_label_num__}" if __version_label_num__ > 0 else f".{__version_label__}"

# Check for version file or environment variable on module load
file_version = get_version_from_file()
if file_version:
    os.environ["DOC_FORGE_VERSION"] = file_version
    update_version_from_env()
else:
    update_version_from_env()

if __name__ == "__main__":
    print(f"Doc Forge v{VERSION} (PEP440: {PEP440_VERSION})")
    print(f"Version components: {get_version_tuple()}")
    print(f"Version info: {get_version_info()}")
