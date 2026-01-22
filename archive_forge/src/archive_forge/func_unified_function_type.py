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
def unified_function_type(numba_types, require_precise=True):
    """Returns a unified Numba function type if possible.

    Parameters
    ----------
    numba_types : Sequence of numba Type instances.
    require_precise : bool
      If True, the returned Numba function type must be precise.

    Returns
    -------
    typ : {numba.core.types.Type, None}
      A unified Numba function type. Or ``None`` when the Numba types
      cannot be unified, e.g. when the ``numba_types`` contains at
      least two different Numba function type instances.

    If ``numba_types`` contains a Numba dispatcher type, the unified
    Numba function type will be an imprecise ``UndefinedFunctionType``
    instance, or None when ``require_precise=True`` is specified.

    Specifying ``require_precise=False`` enables unifying imprecise
    Numba dispatcher instances when used in tuples or if-then branches
    when the precise Numba function cannot be determined on the first
    occurrence that is not a call expression.
    """
    from numba.core.errors import NumbaExperimentalFeatureWarning
    if not (isinstance(numba_types, Sequence) and len(numba_types) > 0 and isinstance(numba_types[0], (types.Dispatcher, types.FunctionType))):
        return
    warnings.warn('First-class function type feature is experimental', category=NumbaExperimentalFeatureWarning)
    mnargs, mxargs = (None, None)
    dispatchers = set()
    function = None
    undefined_function = None
    for t in numba_types:
        if isinstance(t, types.Dispatcher):
            mnargs1, mxargs1 = get_nargs_range(t.dispatcher.py_func)
            if mnargs is None:
                mnargs, mxargs = (mnargs1, mxargs1)
            elif not (mnargs, mxargs) == (mnargs1, mxargs1):
                return
            dispatchers.add(t.dispatcher)
            t = t.dispatcher.get_function_type()
            if t is None:
                continue
        if isinstance(t, types.FunctionType):
            if mnargs is None:
                mnargs = mxargs = t.nargs
            elif not mnargs == mxargs == t.nargs:
                return
            if isinstance(t, types.UndefinedFunctionType):
                if undefined_function is None:
                    undefined_function = t
                else:
                    return
                dispatchers.update(t.dispatchers)
            elif function is None:
                function = t
            else:
                assert function == t
        else:
            return
    if require_precise and (function is None or undefined_function is not None):
        return
    if function is not None:
        if undefined_function is not None:
            assert function.nargs == undefined_function.nargs
            function = undefined_function
    elif undefined_function is not None:
        undefined_function.dispatchers.update(dispatchers)
        function = undefined_function
    else:
        function = types.UndefinedFunctionType(mnargs, dispatchers)
    return function