from __future__ import annotations
import warnings
import pytest
from datashader.macros import expand_varargs
import inspect
from numba import jit
@jit(nopython=True, nogil=True)
def function_no_vararg_numba(a, b):
    return a + b