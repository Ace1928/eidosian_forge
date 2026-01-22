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
def test_roll_basic(self):
    pyfunc = roll
    cfunc = jit(nopython=True)(pyfunc)

    def a_variations():
        yield np.arange(7)
        yield np.arange(3 * 4 * 5).reshape(3, 4, 5)
        yield [1.1, 2.2, 3.3]
        yield (True, False, True)
        yield False
        yield 4
        yield (9,)
        yield np.asfortranarray(np.array([[1.1, np.nan], [np.inf, 7.8]]))
        yield np.array([])
        yield ()

    def shift_variations():
        return itertools.chain.from_iterable(((True, False), range(-10, 10)))
    for a in a_variations():
        for shift in shift_variations():
            expected = pyfunc(a, shift)
            got = cfunc(a, shift)
            self.assertPreciseEqual(expected, got)