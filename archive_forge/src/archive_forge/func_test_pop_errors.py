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
def test_pop_errors(self):
    self.disable_leak_check()
    cfunc = jit(nopython=True)(list_pop1)
    with self.assertRaises(IndexError) as cm:
        cfunc(0, 5)
    self.assertEqual(str(cm.exception), 'pop from empty list')
    with self.assertRaises(IndexError) as cm:
        cfunc(1, 5)
    self.assertEqual(str(cm.exception), 'pop index out of range')