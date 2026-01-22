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
def check_slicing2(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    sizes = [5, 40]
    for n in sizes:
        indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
        for start, stop in itertools.product(indices, indices):
            expected = pyfunc(n, start, stop)
            self.assertPreciseEqual(cfunc(n, start, stop), expected)