import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_try_in_prange_reduction(self):

    def udt(n):
        c = 0
        for i in prange(n):
            try:
                c += 1
            except Exception:
                c += 1
        return c
    args = [10]
    expect = udt(*args)
    self.assertEqual(njit(parallel=False)(udt)(*args), expect)
    self.assertEqual(njit(parallel=True)(udt)(*args), expect)