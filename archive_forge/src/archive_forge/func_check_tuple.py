from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def check_tuple(*args):
    expected_tuple = pyfunc(*args)
    got_tuple = cfunc(*args)
    self.assertEqual(len(got_tuple), len(expected_tuple))
    for got, expected in zip(got_tuple, expected_tuple):
        check_result(got, expected)