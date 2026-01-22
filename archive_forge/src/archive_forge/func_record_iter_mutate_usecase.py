import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def record_iter_mutate_usecase(iterable):
    for x in iterable:
        x.a = x.a + x.b