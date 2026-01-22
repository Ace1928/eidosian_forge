from __future__ import annotations
import contextlib
import datetime
import errno
import hashlib
import importlib
import importlib.util
import inspect
import locale
import os
import os.path
import re
import sys
import types
from types import ModuleType
from typing import (
from coverage import env
from coverage.exceptions import CoverageException
from coverage.types import TArc
from coverage.exceptions import *   # pylint: disable=wildcard-import
def human_sorted(strings: Iterable[str]) -> list[str]:
    """Sort the given iterable of strings the way that humans expect.

    Numeric components in the strings are sorted as numbers.

    Returns the sorted list.

    """
    return sorted(strings, key=_human_key)