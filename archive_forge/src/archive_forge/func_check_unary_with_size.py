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
def check_unary_with_size(self, pyfunc, precise=True):
    cfunc = jit(nopython=True)(pyfunc)
    for n in [0, 3, 16, 70, 400]:
        eq = self.assertPreciseEqual if precise else self.assertEqual
        eq(cfunc(n), pyfunc(n))