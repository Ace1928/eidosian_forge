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
def test_trim_zeros_exceptions(self):
    self.disable_leak_check()
    cfunc = jit(nopython=True)(np_trim_zeros)
    with self.assertRaises(TypingError) as raises:
        cfunc(np.array([[1, 2, 3], [4, 5, 6]]))
    self.assertIn('array must be 1D', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(3)
    self.assertIn('The first argument must be an array', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc({0, 1, 2})
    self.assertIn('The first argument must be an array', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(np.array([0, 1, 2]), 1)
    self.assertIn('The second argument must be a string', str(raises.exception))