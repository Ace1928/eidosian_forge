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
@unittest.skip('dynamic range checking not implemented')
def test_basic50(self):
    """2 args, standard_indexing OOB"""

    def kernel(a, b):
        return a[0, 1] + b[0, 15]