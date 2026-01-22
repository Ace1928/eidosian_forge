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
def get_nargs_range(pyfunc):
    """Return the minimal and maximal number of Python function
    positional arguments.
    """
    sig = pysignature(pyfunc)
    min_nargs = 0
    max_nargs = 0
    for p in sig.parameters.values():
        max_nargs += 1
        if p.default == inspect._empty:
            min_nargs += 1
    return (min_nargs, max_nargs)