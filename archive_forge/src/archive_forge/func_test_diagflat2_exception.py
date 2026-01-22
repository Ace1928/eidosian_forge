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
def test_diagflat2_exception(self):
    pyfunc = diagflat2
    cfunc = njit(pyfunc)
    self.disable_leak_check()
    with self.assertRaises(TypingError) as raises:
        cfunc('abc', 2)
    self.assertIn('The argument "v" must be array-like', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc([1, 2], 'abc')
    self.assertIn('The argument "k" must be an integer', str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc([1, 2], 3.0)
    self.assertIn('The argument "k" must be an integer', str(raises.exception))