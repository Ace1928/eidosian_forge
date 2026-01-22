from functools import partial
import itertools
from itertools import chain, product, starmap
import sys
import numpy as np
from numba import jit, literally, njit, typeof, TypingError
from numba.core import utils, types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.types.functions import _header_lead
import unittest
def test_slice_constructor(self):
    """
        Test the 'happy path' for slice() constructor in nopython mode.
        """
    maxposint = sys.maxsize
    maxnegint = -maxposint - 1
    a = np.arange(10)
    cfunc = jit(nopython=True)(slice_constructor)
    cfunc_use = jit(nopython=True)(slice_construct_and_use)
    for args, expected in [((None,), (0, maxposint, 1)), ((5,), (0, 5, 1)), ((None, None), (0, maxposint, 1)), ((1, None), (1, maxposint, 1)), ((None, 2), (0, 2, 1)), ((1, 2), (1, 2, 1)), ((None, None, 3), (0, maxposint, 3)), ((None, 2, 3), (0, 2, 3)), ((1, None, 3), (1, maxposint, 3)), ((1, 2, 3), (1, 2, 3)), ((None, None, -1), (maxposint, maxnegint, -1)), ((10, None, -1), (10, maxnegint, -1)), ((None, 5, -1), (maxposint, 5, -1)), ((10, 5, -1), (10, 5, -1))]:
        got = cfunc(*args)
        self.assertPreciseEqual(got, expected)
        usage = slice_construct_and_use(args, a)
        cusage = cfunc_use(args, a)
        self.assertPreciseEqual(usage, cusage)