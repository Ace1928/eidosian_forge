"""
File generation utilities for Eidosian repositories.

This module handles the creation and population of files with structured content.
"""
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging


def write_file(path: Path, content: str, overwrite: bool = True) -> bool:
    """
    Write content to a file, creating parent directories if needed.
    
    Args:
        path: Path to the file
        content: Content to write
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if file was written, False if skipped
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if file exists and we're not overwriting
    if path.exists() and not overwrite:
        logging.debug(f"Skipping existing file: {path}")
        return False
        
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    logging.debug(f"Created file: {path}")
    return True


def generate_files(
    base_path: Path, 
    files_dict: Dict[str, str],
    executable: Optional[list] = None,
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Generate multiple files from a dictionary mapping paths to content.
    
    Args:
        base_path: Base directory for file paths
        files_dict: Dictionary mapping relative paths to file content
        executable: List of paths that should be made executable
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with generation results
    """
    created_files = []
    skipped_files = []
    executable = executable or []
    
    for rel_path, content in files_dict.items():
        file_path = base_path / rel_path
        result = write_file(file_path, content, overwrite)
        
        if result:
            created_files.append(rel_path)
            
            # Make file executable if needed
            if rel_path in executable:
                file_path.chmod(file_path.stat().st_mode | 0o111)  # Add executable bit
        else:
            skipped_files.append(rel_path)
    
    return {
        "success": True,
        "created_files": created_files,
        "skipped_files": skipped_files,
        "base_path": str(base_path)
    }