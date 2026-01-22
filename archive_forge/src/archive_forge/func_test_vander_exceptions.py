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
def test_vander_exceptions(self):
    pyfunc = vander
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    x = np.arange(5) - 0.5

    def _check_n(N):
        with self.assertTypingError() as raises:
            cfunc(x, N=N)
        self.assertIn('Second argument N must be None or an integer', str(raises.exception))
    for N in (1.1, True, np.inf, [1, 2]):
        _check_n(N)
    with self.assertRaises(ValueError) as raises:
        cfunc(x, N=-1)
    self.assertIn('Negative dimensions are not allowed', str(raises.exception))

    def _check_1d(x):
        with self.assertRaises(ValueError) as raises:
            cfunc(x)
        self.assertEqual('x must be a one-dimensional array or sequence.', str(raises.exception))
    x = np.arange(27).reshape((3, 3, 3))
    _check_1d(x)
    x = ((2, 3), (4, 5))
    _check_1d(x)