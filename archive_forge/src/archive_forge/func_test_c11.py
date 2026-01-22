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
def test_c11(self):

    def bar(x):
        x[:] = x[::-1]
        return x
    r = [[1, 2, 3], [4, 5, 6]]
    self.compile_and_test(bar, r)