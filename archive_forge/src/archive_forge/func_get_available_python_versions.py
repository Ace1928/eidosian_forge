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
@cache
def get_available_python_versions() -> dict[str, str]:
    """Return a dictionary indicating which supported Python versions are available."""
    return dict(((version, path) for version, path in ((version, find_python(version, required=False)) for version in SUPPORTED_PYTHON_VERSIONS) if path))