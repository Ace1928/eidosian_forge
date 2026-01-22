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
def test_np_append_exceptions(self):
    pyfunc = append
    cfunc = jit(nopython=True)(pyfunc)
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    values = np.array([[7, 8, 9]])
    axis = 0
    with self.assertRaises(TypingError) as raises:
        cfunc(None, values, axis)
    self.assertIn('The first argument "arr" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(arr, None, axis)
    self.assertIn('The second argument "values" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc(arr, values, axis=0.0)
    self.assertIn('The third argument "axis" must be an integer', str(raises.exception))
    self.disable_leak_check()