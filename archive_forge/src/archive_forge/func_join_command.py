import os
import shlex
import subprocess
import sys
from typing import Any, List, Optional, Union
from ..compat import is_windows
from ..errors import Errors
def join_command(command: List[str]) -> str:
    """Join a command using shlex. shlex.join is only available for Python 3.8+,
    so we're using a workaround here.

    command (List[str]): The command to join.
    RETURNS (str): The joined command
    """
    return ' '.join((shlex.quote(cmd) for cmd in command))