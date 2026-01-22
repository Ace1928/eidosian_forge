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
def dollar_replace(match: re.Match[str]) -> str:
    """Called for each $replacement."""
    word = next((g for g in match.group(*dollar_groups) if g))
    if word == '$':
        return '$'
    elif word in variables:
        return variables[word]
    elif match['strict']:
        msg = f'Variable {word} is undefined: {text!r}'
        raise CoverageException(msg)
    else:
        return match['defval']