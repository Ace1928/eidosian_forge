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
def test_vhdsplit_exception(self):
    for f, mindim, name in [(vsplit, 2, 'vsplit'), (hsplit, 1, 'hsplit'), (dsplit, 3, 'dsplit')]:
        cfunc = jit(nopython=True)(f)
        self.disable_leak_check()
        with self.assertRaises(TypingError) as raises:
            cfunc(1, 2)
        self.assertIn('The argument "ary" must be an array', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc('abc', 2)
        self.assertIn('The argument "ary" must be an array', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc(np.array([[1, 2, 3, 4], [1, 2, 3, 4]]), 'abc')
        self.assertIn('The argument "indices_or_sections" must be int or 1d-array', str(raises.exception))
        with self.assertRaises(ValueError) as raises:
            cfunc(np.array(1), 2)
        self.assertIn(name + ' only works on arrays of ' + str(mindim) + ' or more dimensions', str(raises.exception))