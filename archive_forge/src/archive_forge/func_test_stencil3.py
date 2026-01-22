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
def test_stencil3(self):
    """Tests whether a non-zero optional cval argument to the stencil
        decorator works.  Also tests integer result type.
        """

    def test_seq(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = stencil3_kernel(A)
        return B
    test_njit = njit(test_seq)
    test_par = njit(test_seq, parallel=True)
    n = 5
    seq_res = test_seq(n)
    njit_res = test_njit(n)
    par_res = test_par(n)
    self.assertTrue(seq_res[0, 0] == 1.0 and seq_res[4, 4] == 1.0)
    self.assertTrue(njit_res[0, 0] == 1.0 and njit_res[4, 4] == 1.0)
    self.assertTrue(par_res[0, 0] == 1.0 and par_res[4, 4] == 1.0)