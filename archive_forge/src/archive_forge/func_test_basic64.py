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
def test_basic64(self):
    """1 arg that uses standard_indexing"""

    def kernel(a):
        return a[0, 0]
    a = np.arange(12.0).reshape(3, 4)
    self.check_exceptions(kernel, a, options={'standard_indexing': 'a'}, expected_exception=[ValueError, NumbaValueError])