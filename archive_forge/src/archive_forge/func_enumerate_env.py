import json
import os
import shlex
import sys
from contextlib import contextmanager
from subprocess import Popen
from typing import Any, Dict, IO, Iterator, List
from .main import dotenv_values, set_key, unset_key
from .version import __version__
def enumerate_env():
    """
    Return a path for the ${pwd}/.env file.

    If pwd does not exist, return None.
    """
    try:
        cwd = os.getcwd()
    except FileNotFoundError:
        return None
    path = os.path.join(cwd, '.env')
    return path