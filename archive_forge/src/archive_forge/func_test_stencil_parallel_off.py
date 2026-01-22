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
@skip_unsupported
def test_stencil_parallel_off(self):
    """Tests 1D numba.stencil calls without parallel translation
           turned off.
        """

    def test_impl(A):
        return numba.stencil(lambda a: 0.3 * (a[-1] + a[0] + a[1]))(A)
    cpfunc = self.compile_parallel(test_impl, (numba.float64[:],), stencil=False)
    self.assertNotIn('@do_scheduling', cpfunc.library.get_llvm_str())