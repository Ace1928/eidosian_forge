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
def test_histogram(self):
    pyfunc = histogram
    cfunc = jit(nopython=True)(pyfunc)

    def check(*args):
        pyhist, pybins = pyfunc(*args)
        chist, cbins = cfunc(*args)
        self.assertPreciseEqual(pyhist, chist)
        self.assertPreciseEqual(pybins, cbins, prec='double', ulps=2)

    def check_values(values):
        bins = np.float64([1, 3, 4.5, 8])
        check(values, bins)
        check(values.reshape((3, 4)), bins)
        check(values, 7)
        check(values, 7, (1.0, 13.5))
        check(values)
    values = np.float64((0, 0.99, 1, 4.4, 4.5, 7, 8, 9, 9.5, 42.5, -1.0, -0.0))
    assert len(values) == 12
    self.rnd.shuffle(values)
    check_values(values)