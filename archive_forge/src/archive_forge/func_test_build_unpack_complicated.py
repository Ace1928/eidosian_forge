import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_build_unpack_complicated(self):

    def check(p):

        def pyfunc(a):
            z = [1, 2]
            return ((*a, *(*a, a), *(a, (*(a, (1, 2), *(3,), *a), (a, 1, (2, 3), *a, 1), (1,))), *(z.append(4), z.extend(a))), z)
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(p), pyfunc(p))
    check((10, 20))