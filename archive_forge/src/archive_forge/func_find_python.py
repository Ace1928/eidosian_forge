from __future__ import annotations
import abc
import collections.abc as c
import enum
import fcntl
import importlib.util
import inspect
import json
import keyword
import os
import platform
import pkgutil
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import functools
import shlex
import typing as t
import warnings
from struct import unpack, pack
from termios import TIOCGWINSZ
from .locale_util import (
from .encoding import (
from .io import (
from .thread import (
from .constants import (
def find_python(version: str, path: t.Optional[str]=None, required: bool=True) -> t.Optional[str]:
    """
    Find and return the full path to the specified Python version.
    If required, an exception will be raised not found.
    If not required, None will be returned if not found.
    """
    version_info = str_to_version(version)
    if not path and version_info == sys.version_info[:len(version_info)]:
        python_bin = sys.executable
    else:
        python_bin = find_executable('python%s' % version, path=path, required=required)
    return python_bin