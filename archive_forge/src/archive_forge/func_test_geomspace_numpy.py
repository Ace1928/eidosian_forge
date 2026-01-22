import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_geomspace_numpy(self):
    cfunc2 = jit(nopython=True)(geomspace2)
    cfunc3 = jit(nopython=True)(geomspace3)
    pfunc3 = geomspace3
    y = cfunc2(1, 1000000.0)
    self.assertEqual(len(y), 50)
    y = cfunc3(1, 1000000.0, num=100)
    self.assertEqual(y[-1], 10 ** 6)
    y = cfunc3(1, 1000000.0, num=7)
    self.assertPreciseEqual(y, pfunc3(1, 1000000.0, num=7))
    y = cfunc3(8, 2, num=3)
    self.assertPreciseEqual(y, pfunc3(8, 2, num=3))
    self.assertTrue([x == 0 for x in y.imag])
    y = cfunc3(-1, -100, num=3)
    self.assertPreciseEqual(y, pfunc3(-1, -100, num=3))
    self.assertTrue([x == 0 for x in y.imag])
    y = cfunc3(-100, -1, num=3)
    self.assertPreciseEqual(y, pfunc3(-100, -1, num=3))
    self.assertTrue([x == 0 for x in y.imag])
    start = 0.3
    stop = 20.3
    y = cfunc3(start, stop, num=1)
    self.assertPreciseEqual(y[0], start)
    y = cfunc3(start, stop, num=3)
    self.assertPreciseEqual(y[0], start)
    self.assertPreciseEqual(y[-1], stop)
    with np.errstate(invalid='ignore'):
        y = cfunc3(-3, 3, num=4)
    self.assertPreciseEqual(y[0], -3.0)
    self.assertTrue(np.isnan(y[1:-1]).all())
    self.assertPreciseEqual(y[3], 3.0)
    y = cfunc3(1j, 16j, num=5)
    self.assertPreciseEqual(y, pfunc3(1j, 16j, num=5), abs_tol=1e-14)
    self.assertTrue([x == 0 for x in y.real])
    y = cfunc3(-4j, -324j, num=5)
    self.assertPreciseEqual(y, pfunc3(-4j, -324j, num=5), abs_tol=1e-13)
    self.assertTrue([x == 0 for x in y.real])
    y = cfunc3(1 + 1j, 1000 + 1000j, num=4)
    self.assertPreciseEqual(y, pfunc3(1 + 1j, 1000 + 1000j, num=4), abs_tol=1e-13)
    y = cfunc3(-1 + 1j, -1000 + 1000j, num=4)
    self.assertPreciseEqual(y, pfunc3(-1 + 1j, -1000 + 1000j, num=4), abs_tol=1e-13)
    y = cfunc3(-1 + 0j, 1 + 0j, num=3)
    self.assertPreciseEqual(y, pfunc3(-1 + 0j, 1 + 0j, num=3))
    y = cfunc3(0 + 3j, -3 + 0j, 3)
    self.assertPreciseEqual(y, pfunc3(0 + 3j, -3 + 0j, 3), abs_tol=1e-15)
    y = cfunc3(0 + 3j, 3 + 0j, 3)
    self.assertPreciseEqual(y, pfunc3(0 + 3j, 3 + 0j, 3), abs_tol=1e-15)
    y = cfunc3(-3 + 0j, 0 - 3j, 3)
    self.assertPreciseEqual(y, pfunc3(-3 + 0j, 0 - 3j, 3), abs_tol=1e-15)
    y = cfunc3(0 + 3j, -3 + 0j, 3)
    self.assertPreciseEqual(y, pfunc3(0 + 3j, -3 + 0j, 3), abs_tol=1e-15)
    y = cfunc3(-2 - 3j, 5 + 7j, 7)
    self.assertPreciseEqual(y, pfunc3(-2 - 3j, 5 + 7j, 7), abs_tol=1e-14)
    y = cfunc3(3j, -5, 2)
    self.assertPreciseEqual(y, pfunc3(3j, -5, 2))
    y = cfunc3(-5, 3j, 2)
    self.assertPreciseEqual(y, pfunc3(-5, 3j, 2))