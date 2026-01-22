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
def unify_function_types(numba_types):
    """Return a normalized tuple of Numba function types so that

        Tuple(numba_types)

    becomes

        UniTuple(dtype=<unified function type>, count=len(numba_types))

    If the above transformation would be incorrect, return the
    original input as given. For instance, if the input tuple contains
    types that are not function or dispatcher type, the transformation
    is considered incorrect.
    """
    dtype = unified_function_type(numba_types)
    if dtype is None:
        return numba_types
    return (dtype,) * len(numba_types)