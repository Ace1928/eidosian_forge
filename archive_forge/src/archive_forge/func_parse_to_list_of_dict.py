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
def parse_to_list_of_dict(pattern: str, value: str) -> list[dict[str, str]]:
    """Parse lines from the given value using the specified pattern and return the extracted list of key/value pair dictionaries."""
    matched = []
    unmatched = []
    for line in value.splitlines():
        match = re.search(pattern, line)
        if match:
            matched.append(match.groupdict())
        else:
            unmatched.append(line)
    if unmatched:
        raise Exception('Pattern "%s" did not match values:\n%s' % (pattern, '\n'.join(unmatched)))
    return matched