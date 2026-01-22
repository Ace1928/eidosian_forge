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
def test_unwrap_exception(self):
    cfunc = njit(unwrap)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as e:
        cfunc('abc')
    self.assertIn('The argument "p" must be array-like', str(e.exception))
    with self.assertRaises(TypingError) as e:
        cfunc(np.array([1, 2]), 'abc')
    self.assertIn('The argument "discont" must be a scalar', str(e.exception))
    with self.assertRaises(TypingError) as e:
        cfunc(np.array([1, 2]), 3, period='abc')
    self.assertIn('The argument "period" must be a scalar', str(e.exception))
    with self.assertRaises(TypingError) as e:
        cfunc(np.array([1, 2]), 3, axis='abc')
    self.assertIn('The argument "axis" must be an integer', str(e.exception))
    with self.assertRaises(ValueError) as e:
        cfunc(np.array([1, 2]), 3, axis=2)
    self.assertIn('Value for argument "axis" is not supported', str(e.exception))