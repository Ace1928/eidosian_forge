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
def test_cross_exceptions(self):
    pyfunc = np_cross
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertRaises(ValueError) as raises:
        cfunc(np.arange(4), np.arange(3))
    self.assertIn('Incompatible dimensions for cross product', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc(np.array((1, 2)), np.array((3, 4)))
    self.assertIn('Dimensions for both inputs is 2.', str(raises.exception))
    self.assertIn('`cross2d(a, b)` from `numba.np.extensions`.', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc(np.arange(8).reshape((2, 4)), np.arange(6)[::-1].reshape((2, 3)))
    self.assertIn('Incompatible dimensions for cross product', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        cfunc(np.arange(8).reshape((4, 2)), np.arange(8)[::-1].reshape((4, 2)))
    self.assertIn('Dimensions for both inputs is 2', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(set([1, 2, 3]), set([4, 5, 6]))
    self.assertIn('Inputs must be array-like.', str(raises.exception))