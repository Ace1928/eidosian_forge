import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
def exception_dict(self, **kwargs):
    d = dict()
    d['pyStencil'] = None
    d['stencil'] = None
    d['njit'] = None
    d['parfor'] = None
    for k, v in kwargs.items():
        d[k] = v
    return d