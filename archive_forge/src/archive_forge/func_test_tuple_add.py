import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_tuple_add(self):

    def pyfunc(x):
        a = np.arange(3)
        return (a,) + (x,)
    cfunc = jit(nopython=True)(pyfunc)
    x = 123
    expect_a, expect_x = pyfunc(x)
    got_a, got_x = cfunc(x)
    np.testing.assert_equal(got_a, expect_a)
    self.assertEqual(got_x, expect_x)