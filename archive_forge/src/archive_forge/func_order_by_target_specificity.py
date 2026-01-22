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
def order_by_target_specificity(target, templates, fnkey=''):
    """This orders the given templates from most to least specific against the
    current "target". "fnkey" is an indicative typing key for use in the
    exception message in the case that there's no usable templates for the
    current "target".
    """
    if templates == []:
        return []
    from numba.core.target_extension import target_registry
    DEFAULT_TARGET = 'generic'
    usable = []
    for ix, temp_cls in enumerate(templates):
        md = getattr(temp_cls, 'metadata', {})
        hw = md.get('target', DEFAULT_TARGET)
        if hw is not None:
            hw_clazz = target_registry[hw]
            if target.inherits_from(hw_clazz):
                usable.append((temp_cls, hw_clazz, ix))

    def key(x):
        return target.__mro__.index(x[1])
    order = [x[0] for x in sorted(usable, key=key)]
    if not order:
        msg = f"Function resolution cannot find any matches for function '{fnkey}' for the current target: '{target}'."
        from numba.core.errors import UnsupportedError
        raise UnsupportedError(msg)
    return order