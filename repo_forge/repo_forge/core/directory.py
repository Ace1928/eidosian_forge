"""
Directory structure creation for Eidosian repositories.

This module handles the precise creation of directory hierarchies following
Eidosian principles of structural perfection.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import concurrent.futures
from functools import partial

from ..constants.paths import (
    CORE_DIRECTORIES,
    LANGUAGE_DIRECTORIES,
    SCRIPT_DIRECTORIES,
    TEST_DIRECTORIES,
    BENCHMARK_DIRECTORIES,
    EXAMPLE_DIRECTORIES,
    CI_DIRECTORIES,
    GITHUB_DIRECTORIES,
    SHARED_DIRECTORIES,
    CONFIG_DIRECTORIES,
    TOOL_DIRECTORIES
)


def create_directory(path: Path) -> None:
    """Create a directory and its parents if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)


def create_directories_in_parallel(base_path: Path, directories: List[str]) -> None:
    """Create multiple directories in parallel for optimal performance."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        create_dir = partial(create_directory)
        paths = [base_path / directory for directory in directories]
        list(executor.map(create_dir, paths))


def create_directory_structure(base_path: Path, languages: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create the complete directory structure for an Eidosian monorepo.
    
    Args:
        base_path: Base directory where the repo will be created
        languages: List of programming languages to support
        
    Returns:
        Dictionary with creation results
    """
    if languages is None:
        languages = ["python", "nodejs", "go", "rust"]
        
    logging.debug(f"Creating core directories at {base_path}")
    
    # Create top-level directories in parallel for efficiency
    create_directories_in_parallel(base_path, CORE_DIRECTORIES)
    
    # Create projects directory with language-specific subdirectories
    projects_path = base_path / "projects"
    for language in languages:
        dir_config = LANGUAGE_DIRECTORIES.get(language, {})
        if not dir_config:
            continue
            
        # Create main project structure
        project_path = projects_path / f"{language}_project"
        create_directory(project_path / "src")
        create_directory(project_path / "tests")
        
        # Create language-specific subdirectories
        for subdir in dir_config.get("structure", []):
            create_directory(project_path / "src" / f"{language}_project" / subdir)
            
        # Touch required files
        for file_path in dir_config.get("files", []):
            (project_path / "src" / f"{language}_project" / file_path).touch()
    
    # Create script categories
    for script_dir in SCRIPT_DIRECTORIES:
        create_directory(base_path / "scripts" / script_dir)
    
    # Create test categories
    for test_dir in TEST_DIRECTORIES:
        create_directory(base_path / "tests" / test_dir)
    
    # Create benchmark categories
    for bench_dir in BENCHMARK_DIRECTORIES:
        create_directory(base_path / "benchmarks" / bench_dir)
    
    # Create example categories
    for example_dir in EXAMPLE_DIRECTORIES:
        create_directory(base_path / "examples" / example_dir)
    
    # Create CI and GitHub directories
    for ci_dir in CI_DIRECTORIES:
        create_directory(base_path / "ci" / ci_dir)
        
    for github_dir in GITHUB_DIRECTORIES:
        create_directory(base_path / ".github" / github_dir)
    
    # Create shared cross-language directories
    for shared_dir in SHARED_DIRECTORIES:
        create_directory(base_path / "shared" / shared_dir)
    
    # Create config directories
    for config_dir in CONFIG_DIRECTORIES:
        create_directory(base_path / "config" / config_dir)
    
    # Create tool directories
    for tool_dir in TOOL_DIRECTORIES:
        create_directory(base_path / "tools" / tool_dir)
    
    # Create empty placeholder files to maintain directory structure in git
    for dirpath, dirnames, _ in os.walk(str(base_path)):
        if not os.path.basename(dirpath).startswith('.') and not os.listdir(dirpath):
            Path(dirpath, ".gitkeep").touch()
    
    return {
        "success": True,
        "base_path": str(base_path),
        "languages": languages
    }