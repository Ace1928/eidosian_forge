import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_np_ndindex(self):
    func = np_ndindex
    cfunc = njit((types.int32, types.int32))(func)
    self.assertPreciseEqual(cfunc(3, 4), func(3, 4))
    self.assertPreciseEqual(cfunc(3, 0), func(3, 0))
    self.assertPreciseEqual(cfunc(0, 3), func(0, 3))
    self.assertPreciseEqual(cfunc(0, 0), func(0, 0))