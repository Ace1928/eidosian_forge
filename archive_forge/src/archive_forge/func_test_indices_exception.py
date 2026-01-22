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
def test_indices_exception(self):
    cfunc = njit(np_indices)
    self.disable_leak_check()
    errmsg = 'The argument "dimensions" must be a tuple of integers'
    with self.assertRaises(TypingError) as raises:
        cfunc('abc')
    self.assertIn(errmsg, str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc((2.0, 3.0))
    self.assertIn(errmsg, str(raises.exception))
    with self.assertRaises(TypingError) as raises:
        cfunc((2, 3.0))
    self.assertIn(errmsg, str(raises.exception))