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
def exc_one_line(exc: Exception) -> str:
    """Get a one-line summary of an exception, including class name and message."""
    lines = traceback.format_exception_only(type(exc), exc)
    return '|'.join((l.rstrip() for l in lines))