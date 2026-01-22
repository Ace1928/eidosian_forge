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
def test_c08(self):
    self.disable_leak_check()

    def bar(x):
        x[5] = 7
        return x
    r = [1, 2, 3]
    cfunc = jit(nopython=True)(bar)
    with self.assertRaises(IndexError) as raises:
        cfunc(r)
    self.assertIn('setitem out of range', str(raises.exception))