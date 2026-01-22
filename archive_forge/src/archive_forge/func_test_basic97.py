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
@unittest.skip('not yet supported')
def test_basic97(self):
    """ 2D slice and index. """

    def kernel(a):
        return np.median(a[-1:2, 3])