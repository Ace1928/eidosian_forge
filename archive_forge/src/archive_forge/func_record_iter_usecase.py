import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def record_iter_usecase(iterable):
    res = 0.0
    for x in iterable:
        res += x.a * x.b
    return res