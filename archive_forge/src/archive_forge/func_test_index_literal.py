import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_index_literal(self):

    def pyfunc(tup, idx):
        idx = literally(idx)
        return tup[idx]
    cfunc = njit(pyfunc)
    tup = (4, 3.1, 'sss')
    for i in range(len(tup)):
        self.assertPreciseEqual(cfunc(tup, i), tup[i])