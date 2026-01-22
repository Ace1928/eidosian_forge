from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
@from_generic(pyfuncs_to_use)
def check_err_invalid_args(pyfunc, arr, axes):
    with self.assertRaises((TypeError, TypingError)):
        pyfunc(arr, axes)