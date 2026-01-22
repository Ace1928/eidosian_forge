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
def test_slice_from_constant(self):
    test_tuple = (1, 2, 3, 4)
    for ts in itertools.product([None, 1, 2, 3], [None, 1, 2, 3], [None, 1, 2, -1, -2]):
        ts = slice(*ts)

        @jit(nopython=True)
        def test_fn():
            return test_tuple[ts]
        self.assertEqual(test_fn(), test_fn.py_func())