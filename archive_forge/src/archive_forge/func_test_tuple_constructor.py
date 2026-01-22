import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_tuple_constructor(self):

    def check(pyfunc, arg):
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(arg), pyfunc(arg))
    check(lambda _: tuple(), ())
    check(lambda a: tuple(a), (4, 5))
    check(lambda a: tuple(a), (4, 5.5))