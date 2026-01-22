"""
Shared utility functions for the Eidosian Repo Forge.

This module provides common functionality used across the package.
"""
import os
import subprocess
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import logging


def run_command(
    cmd: List[str], 
    cwd: Optional[Union[str, Path]] = None,
    capture_output: bool = False
) -> Dict[str, Any]:
    """
    Run a shell command safely.
    
    Args:
        cmd: Command and arguments as a list
        cwd: Working directory for the command
        capture_output: Whether to capture and return output
        
    Returns:
        Dictionary with command results
    """
    logging.debug(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=capture_output,
            check=True
        )
        return {
            "success": True,
            "command": cmd,
            "returncode": 0,
            "stdout": result.stdout if capture_output else None,
            "stderr": result.stderr if capture_output else None
        }
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {' '.join(cmd)}")
        return {
            "success": False,
            "command": cmd,
            "returncode": e.returncode,
            "stdout": e.stdout if capture_output else None,
            "stderr": e.stderr if capture_output else None,
            "error": str(e)
        }


def make_executable(file_path: Union[str, Path]) -> bool:
    """
    Make a file executable.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        current_mode = file_path.stat().st_mode
        executable_mode = current_mode | 0o111  # Add executable bit for user, group, others
        file_path.chmod(executable_mode)
        return True
    except Exception as e:
        logging.error(f"Failed to make {file_path} executable: {e}")
        return False


def detect_environment() -> Dict[str, Any]:
    """
    Detect information about the execution environment.
    
    Returns:
        Dictionary with environment information
    """
    import platform
    
    return {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "has_git": bool(subprocess.run(
            ["git", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE).returncode == 0),
        "has_docker": bool(subprocess.run(
            ["docker", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE).returncode == 0),
    }