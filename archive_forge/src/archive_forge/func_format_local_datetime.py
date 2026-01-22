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
def format_local_datetime(dt: datetime.datetime) -> str:
    """Return a string with local timezone representing the date.
    """
    return dt.astimezone().strftime('%Y-%m-%d %H:%M %z')