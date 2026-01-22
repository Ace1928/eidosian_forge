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
def exclude_none_values(data: dict[TKey, t.Optional[TValue]]) -> dict[TKey, TValue]:
    """Return the provided dictionary with any None values excluded."""
    return dict(((key, value) for key, value in data.items() if value is not None))