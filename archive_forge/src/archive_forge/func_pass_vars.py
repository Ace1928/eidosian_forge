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
def pass_vars(required: c.Collection[str], optional: c.Collection[str]) -> dict[str, str]:
    """Return a filtered dictionary of environment variables based on the current environment."""
    env = {}
    for name in required:
        if name not in os.environ:
            raise MissingEnvironmentVariable(name)
        env[name] = os.environ[name]
    for name in optional:
        if name not in os.environ:
            continue
        env[name] = os.environ[name]
    return env