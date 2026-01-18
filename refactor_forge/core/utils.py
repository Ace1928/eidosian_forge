"""
Utility Functions - Common Operations 🛠️

Provides essential utility functions used throughout the Eidosian Refactor system.
Each function is atomic, focused, and optimized for its specific purpose.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# ╭──────────────────────────────────────────────────────╮
# │  📝 String Processing - Text Manipulation            │
# ╰──────────────────────────────────────────────────────╯

def to_snake_case(text: str) -> str:
    """Convert text to snake_case with precision and elegance.
    
    Args:
        text: Input text to convert
        
    Returns:
        Text in snake_case format
    """
    # Handle camelCase and PascalCase
    s1 = re.sub(r'([A-Z])', r'_\1', text)
    # Handle spaces, hyphens, etc.
    s2 = re.sub(r'[ \-]+', '_', s1)
    # Handle consecutive underscores and ensure lowercase
    return re.sub(r'_+', '_', s2).strip('_').lower()


def to_pascal_case(text: str) -> str:
    """Convert text to PascalCase with elegant capitalization.
    
    Args:
        text: Input text to convert
        
    Returns:
        Text in PascalCase format
    """
    # First to snake for normalization
    snake = to_snake_case(text)
    # Then to pascal
    return ''.join(word.capitalize() for word in snake.split('_'))


def extract_indentation(line: str) -> str:
    """Extract leading indentation from a line of text.
    
    Args:
        line: Line of text
        
    Returns:
        Leading indentation (spaces/tabs)
    """
    match = re.match(r'^(\s*)', line)
    return match.group(1) if match else ''

# ╭──────────────────────────────────────────────────────╮
# │  📂 File Operations - Path Handling                  │
# ╰──────────────────────────────────────────────────────╯

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def derive_package_name(source_path: Union[str, Path]) -> str:
    """Derive a suitable package name from a source file path.
    
    Args:
        source_path: Path to source file
        
    Returns:
        Snake-cased package name
    """
    stem = Path(source_path).stem
    return to_snake_case(stem)


def derive_output_dir(source_path: Union[str, Path], package_name: Optional[str] = None) -> Path:
    """Derive an appropriate output directory from source path.
    
    Args:
        source_path: Path to source file
        package_name: Optional package name
        
    Returns:
        Path to output directory
    """
    source_dir = Path(source_path).parent
    pkg_name = package_name or derive_package_name(source_path)
    return source_dir / pkg_name
