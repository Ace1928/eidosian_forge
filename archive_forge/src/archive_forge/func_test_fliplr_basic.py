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
def test_fliplr_basic(self):
    pyfunc = fliplr
    cfunc = jit(nopython=True)(pyfunc)

    def a_variations():
        yield np.arange(10).reshape(5, 2)
        yield np.arange(20).reshape(5, 2, 2)
        yield ((1, 2),)
        yield ([1, 2], [3, 4])
    for a in a_variations():
        expected = pyfunc(a)
        got = cfunc(a)
        self.assertPreciseEqual(expected, got)
    with self.assertRaises(TypingError) as raises:
        cfunc('abc')
    self.assertIn('Cannot np.fliplr on %s type' % types.unicode_type, str(raises.exception))