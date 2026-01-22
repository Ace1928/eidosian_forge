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
def test_with_out(n):
    A = np.arange(n ** 2).reshape((n, n))
    B = np.zeros(n ** 2).reshape((n, n))
    B = stencil1_kernel(A, out=B)
    return B