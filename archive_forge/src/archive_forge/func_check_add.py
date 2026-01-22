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
def check_add(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    sizes = [0, 3, 50, 300]
    for m, n in itertools.product(sizes, sizes):
        expected = pyfunc(m, n)
        self.assertPreciseEqual(cfunc(m, n), expected)