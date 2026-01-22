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
def test_index3(self):
    self.disable_leak_check()
    pyfunc = list_index3
    cfunc = jit(nopython=True)(pyfunc)
    n = 16
    for v in (0, 1, 5, 10, 99999999):
        indices = [0, 1, n - 2, n - 1, n + 1, -1, -2, -n + 3, -n - 1]
        for start, stop in itertools.product(indices, indices):
            self.check_index_result(pyfunc, cfunc, (16, v, start, stop))