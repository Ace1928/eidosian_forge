import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
def check_func_out(self, pyfunc, cfunc, args, out):
    copier = self._aligned_copy if _is_armv7l else np.copy
    with self.assertNoNRTLeak():
        expected = copier(out)
        got = copier(out)
        self.assertIs(pyfunc(*args, out=expected), expected)
        self.assertIs(cfunc(*args, out=got), got)
        self.assertPreciseEqual(got, expected, ignore_sign_on_zero=True)
        del got, expected