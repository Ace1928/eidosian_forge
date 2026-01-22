import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def home_dir() -> str:
    """Get the user's home directory.

    Uses the same logic as the Docker Engine client - use %USERPROFILE% on Windows,
    $HOME/getuid on POSIX.
    """
    if IS_WINDOWS_PLATFORM:
        return os.environ.get('USERPROFILE', '')
    else:
        return os.path.expanduser('~')