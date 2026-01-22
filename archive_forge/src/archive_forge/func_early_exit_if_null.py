import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def early_exit_if_null(builder, stack, obj):
    """
    A convenience wrapper for :func:`early_exit_if`, for the common case where
    the CPython API indicates an error by returning ``NULL``.
    """
    return early_exit_if(builder, stack, is_null(builder, obj))