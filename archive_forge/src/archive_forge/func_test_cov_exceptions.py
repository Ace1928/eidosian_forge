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
@needs_blas
def test_cov_exceptions(self):
    pyfunc = cov
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()

    def _check_m(m):
        with self.assertTypingError() as raises:
            cfunc(m)
        self.assertIn('m has more than 2 dimensions', str(raises.exception))
    m = np.ones((5, 6, 7))
    _check_m(m)
    m = ((((1, 2, 3), (2, 2, 2)),),)
    _check_m(m)
    m = [[[5, 6, 7]]]
    _check_m(m)

    def _check_y(m, y):
        with self.assertTypingError() as raises:
            cfunc(m, y=y)
        self.assertIn('y has more than 2 dimensions', str(raises.exception))
    m = np.ones((5, 6))
    y = np.ones((5, 6, 7))
    _check_y(m, y)
    m = np.array((1.1, 2.2, 1.1))
    y = (((1.2, 2.2, 2.3),),)
    _check_y(m, y)
    m = np.arange(3)
    y = np.arange(4)
    with self.assertRaises(ValueError) as raises:
        cfunc(m, y=y)
    self.assertIn('m and y have incompatible dimensions', str(raises.exception))
    m = np.array([-2.1, -1, 4.3]).reshape(1, 3)
    with self.assertRaises(RuntimeError) as raises:
        cfunc(m)
    self.assertIn('2D array containing a single row is unsupported', str(raises.exception))