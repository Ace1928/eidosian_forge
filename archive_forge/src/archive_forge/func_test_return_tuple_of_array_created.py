import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def test_return_tuple_of_array_created(self):

    def pyfunc(x):
        y = np.empty(x.size)
        for i in range(y.size):
            y[i] = x[i] + 1
        out = (y, y)
        return out
    cfunc = nrtjit(pyfunc)
    x = np.random.random(5)
    expected_x, expected_y = pyfunc(x)
    got_x, got_y = cfunc(x)
    np.testing.assert_equal(expected_x, got_x)
    np.testing.assert_equal(expected_y, got_y)
    self.assertEqual(2, sys.getrefcount(got_y))
    self.assertEqual(2, sys.getrefcount(got_y))