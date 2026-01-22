from __future__ import annotations
import hashlib
import ntpath
import os
import os.path
import posixpath
import re
import sys
from typing import Callable, Iterable
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted, isolate_module, join_regex
def python_reported_file(filename: str) -> str:
    """Return the string as Python would describe this file name."""
    if env.PYBEHAVIOR.report_absolute_files:
        filename = os.path.abspath(filename)
    return filename