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
def check_mul(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    for n in [0, 3, 50, 300]:
        for v in [1, 2, 3, 0, -1, -42]:
            expected = pyfunc(n, v)
            self.assertPreciseEqual(cfunc(n, v), expected)