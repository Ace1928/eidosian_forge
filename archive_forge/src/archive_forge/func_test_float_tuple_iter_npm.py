import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def test_float_tuple_iter_npm(self):
    self.test_float_tuple_iter(flags=no_pyobj_flags)