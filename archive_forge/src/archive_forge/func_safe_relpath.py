import atexit
import builtins
import functools
import inspect
import os
import operator
import timeit
import math
import sys
import traceback
import weakref
import warnings
import threading
import contextlib
import typing as _tp
from types import ModuleType
from importlib import import_module
import numpy as np
from inspect import signature as pysignature # noqa: F401
from inspect import Signature as pySignature # noqa: F401
from inspect import Parameter as pyParameter # noqa: F401
from numba.core.config import (PYVERSION, MACHINE_BITS, # noqa: F401
from numba.core import config
from numba.core import types
from collections.abc import Mapping, Sequence, MutableSet, MutableMapping
def safe_relpath(path, start=os.curdir):
    """
    Produces a "safe" relative path, on windows relpath doesn't work across
    drives as technically they don't share the same root.
    See: https://bugs.python.org/issue7195 for details.
    """
    drive_letter = lambda x: os.path.splitdrive(os.path.abspath(x))[0]
    drive_path = drive_letter(path)
    drive_start = drive_letter(start)
    if drive_path != drive_start:
        return os.path.abspath(path)
    else:
        return os.path.relpath(path, start=start)