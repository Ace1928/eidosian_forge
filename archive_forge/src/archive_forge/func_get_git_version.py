import hashlib
import os
import shutil
import sys
from configparser import InterpolationError
from contextlib import contextmanager
from pathlib import Path
from typing import (
import srsly
import typer
from click import NoSuchOption
from click.parser import split_arg_string
from thinc.api import Config, ConfigValidationError, require_gpu
from thinc.util import gpu_is_available
from typer.main import get_command
from wasabi import Printer, msg
from weasel import app as project_cli
from .. import about
from ..compat import Literal
from ..schemas import validate
from ..util import (
def get_git_version(error: str="Could not run 'git'. Make sure it's installed and the executable is available.") -> Tuple[int, int]:
    """Get the version of git and raise an error if calling 'git --version' fails.
    error (str): The error message to show.
    RETURNS (Tuple[int, int]): The version as a (major, minor) tuple. Returns
        (0, 0) if the version couldn't be determined.
    """
    try:
        ret = run_command('git --version', capture=True)
    except:
        raise RuntimeError(error)
    stdout = ret.stdout.strip()
    if not stdout or not stdout.startswith('git version'):
        return (0, 0)
    version = stdout[11:].strip().split('.')
    return (int(version[0]), int(version[1]))