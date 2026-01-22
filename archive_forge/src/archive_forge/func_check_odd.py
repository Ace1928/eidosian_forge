from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_odd(a):
    check(a)
    a = a.reshape((9, 7))
    check(a)
    check(a.T)