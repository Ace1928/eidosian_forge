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
def test_returning_list_of_list(self):

    def pyfunc():
        a = [[np.arange(i)] for i in range(4)]
        return a
    self.compile_and_test(pyfunc)