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
def test_setslice3(self):
    pyfunc = list_setslice3
    cfunc = jit(nopython=True)(pyfunc)
    for n in [10]:
        indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
        steps = [4, 1, -1, 2, -3]
        for start, stop, step in itertools.product(indices, indices, steps):
            expected = pyfunc(n, start, stop, step)
            self.assertPreciseEqual(cfunc(n, start, stop, step), expected)