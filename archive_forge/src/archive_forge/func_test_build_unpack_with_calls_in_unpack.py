import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_build_unpack_with_calls_in_unpack(self):

    def check(p):

        def pyfunc(a):
            z = [1, 2]
            return ((*a, z.append(3), z.extend(a), np.ones(3)), z)
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(p), pyfunc(p))
    check((4, 5))