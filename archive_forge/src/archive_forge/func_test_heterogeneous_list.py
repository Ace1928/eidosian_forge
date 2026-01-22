from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
@expect_reflection_failure
def test_heterogeneous_list(self):

    def pyfunc(x):
        return x[1]
    l1 = [[np.zeros(i) for i in range(5)], [np.ones(i) for i in range(5)]]
    cfunc = jit(nopython=True)(pyfunc)
    l1_got = cfunc(l1)
    self.assertPreciseEqual(pyfunc(l1), l1_got)