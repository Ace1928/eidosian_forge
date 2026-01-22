import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_scalar_tuple(self):
    scalarty = types.float32
    cfunc = njit((scalarty, scalarty))(tuple_return_usecase)
    a = b = 1
    ra, rb = cfunc(a, b)
    self.assertEqual(ra, a)
    self.assertEqual(rb, b)