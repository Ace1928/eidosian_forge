from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
def relevant_environment_display(env: Mapping[str, str]) -> list[tuple[str, str]]:
    """Filter environment variables for a debug display.

    Select variables to display (with COV or PY in the name, or HOME, TEMP, or
    TMP), and also cloak sensitive values with asterisks.

    Arguments:
        env: a dict of environment variable names and values.

    Returns:
        A list of pairs (name, value) to show.

    """
    slugs = {'COV', 'PY'}
    include = {'HOME', 'TEMP', 'TMP'}
    cloak = {'API', 'TOKEN', 'KEY', 'SECRET', 'PASS', 'SIGNATURE'}
    to_show = []
    for name, val in env.items():
        keep = False
        if name in include:
            keep = True
        elif any((slug in name for slug in slugs)):
            keep = True
        if keep:
            if any((slug in name for slug in cloak)):
                val = re.sub('\\w', '*', val)
            to_show.append((name, val))
    return human_sorted_items(to_show)