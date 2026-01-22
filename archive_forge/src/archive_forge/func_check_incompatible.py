import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def check_incompatible(a, b):
    with self.assertRaises(ValueError) as raises:
        cfunc(a, b)
    self.assertIn('operands could not be broadcast together', str(raises.exception))