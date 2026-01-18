"""
Helper functions for integration tests.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def run_cli(args: List[str], input_text: Optional[str] = None) -> Tuple[str, str, int]:
    """
    Run the CLI with the given arguments and input.

    Args:
        args: Command line arguments
        input_text: Optional input text to pipe to stdin

    Returns:
        Tuple of (stdout, stderr, return_code)
    """
    # Find the module path
    module_path = Path(__file__).parent.parent.parent

    cmd = [sys.executable, "-m", "figlet_forge.cli"]
    cmd.extend(args)

    # Set up the process
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input_text else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(module_path),
    )

    # Send input if provided
    stdout, stderr = process.communicate(input=input_text)

    return stdout, stderr, process.returncode
