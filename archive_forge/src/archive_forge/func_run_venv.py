from __future__ import annotations
import collections.abc as c
import json
import os
import pathlib
import sys
import typing as t
from .config import (
from .util import (
from .util_common import (
from .host_configs import (
from .python_requirements import (
def run_venv(args: EnvironmentConfig, run_python: str, system_site_packages: bool, pip: bool, path: str) -> bool:
    """Create a virtual environment using the 'venv' module. Not available on Python 2.x."""
    cmd = [run_python, '-m', 'venv']
    if system_site_packages:
        cmd.append('--system-site-packages')
    if not pip:
        cmd.append('--without-pip')
    cmd.append(path)
    try:
        run_command(args, cmd, capture=True)
    except SubprocessError as ex:
        remove_tree(path)
        if args.verbosity > 1:
            display.error(ex.message)
        return False
    return True