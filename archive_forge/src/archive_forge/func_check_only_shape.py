from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
@from_generic(pyfuncs_to_use)
def check_only_shape(pyfunc, arr, shape, expected_shape):
    self.memory_leak_setup()
    got = generic_run(pyfunc, arr, shape)
    self.assertEqual(got.shape, expected_shape)
    self.assertEqual(got.size, arr.size)
    del got
    self.memory_leak_teardown()