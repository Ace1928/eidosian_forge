"""
Filesystem Manager - File Generation Controller 💾

Controls the generation of files in the output file system,
ensuring proper directory structure and file integrity.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Union


def generate_files(transformation_result: Dict[str, Any], dry_run: bool = False) -> List[str]:
    """Generate files on disk according to transformation results.
    
    Args:
        transformation_result: Results from transformation
        dry_run: If True, only simulate file generation
        
    Returns:
        List of generated file paths
    """
    output_path = Path(transformation_result["output_path"])
    files = transformation_result["files"]
    generated_paths = []
    
    # Create output directory if it doesn't exist
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate each file
    for file_info in files:
        file_path = Path(file_info["path"])
        
        if not dry_run:
            # Ensure parent directories exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_info["content"])
            
            generated_paths.append(str(file_path))
        else:
            generated_paths.append(f"[DRY RUN] Would create: {file_path}")
    
    return generated_paths


def copy_additional_files(
    source_dir: Union[str, Path], 
    dest_dir: Union[str, Path],
    file_patterns: List[str] = ["*.md", "LICENSE*", "setup.py", "pyproject.toml"],
    dry_run: bool = False
) -> List[str]:
    """Copy additional supporting files to output package.
    
    Args:
        source_dir: Source directory to copy from
        dest_dir: Destination directory
        file_patterns: List of file patterns to copy
        dry_run: If True, only simulate file copying
        
    Returns:
        List of copied file paths
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    copied_files = []
    
    # Ensure destination exists
    if not dry_run:
        dest_path.mkdir(parents=True, exist_ok=True)
    
    # Copy matching files
    for pattern in file_patterns:
        for file_path in source_path.glob(pattern):
            if file_path.is_file():
                dest_file = dest_path / file_path.name
                
                if not dry_run:
                    shutil.copy2(file_path, dest_file)
                    copied_files.append(str(dest_file))
                else:
                    copied_files.append(f"[DRY RUN] Would copy: {file_path} to {dest_file}")
    
    return copied_files


def clean_output_directory(
    output_dir: Union[str, Path], 
    keep_patterns: List[str] = ["__init__.py", "README.md"],
    dry_run: bool = False
) -> List[str]:
    """Clean the output directory before generating new files.
    
    Args:
        output_dir: Directory to clean
        keep_patterns: Patterns of files to preserve
        dry_run: If True, only simulate cleaning
        
    Returns:
        List of removed file paths
    """
    output_path = Path(output_dir)
    removed_files = []
    
    # Skip if directory doesn't exist
    if not output_path.exists():
        return []
    
    # Process all files in directory
    for item in output_path.glob("*"):
        # Check if file should be kept
        keep = False
        for pattern in keep_patterns:
            if item.match(pattern):
                keep = True
                break
        
        if not keep:
            if not dry_run:
                if item.is_file():
                    item.unlink()
                    removed_files.append(str(item))
                elif item.is_dir():
                    shutil.rmtree(item)
                    removed_files.append(f"{str(item)}/ (directory)")
            else:
                if item.is_file():
                    removed_files.append(f"[DRY RUN] Would remove: {item}")
                elif item.is_dir():
                    removed_files.append(f"[DRY RUN] Would remove directory: {item}/")
    
    return removed_files
