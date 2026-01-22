import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
def is_buildx_installed() -> bool:
    """Return `True` if docker buildx is installed and working."""
    global _buildx_installed
    if _buildx_installed is not None:
        return _buildx_installed
    if not find_executable('docker'):
        _buildx_installed = False
    else:
        help_output = shell(['buildx', '--help'])
        _buildx_installed = help_output is not None and 'buildx' in help_output
    return _buildx_installed