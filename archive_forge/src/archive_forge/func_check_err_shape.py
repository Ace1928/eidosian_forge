from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
@from_generic(pyfuncs_to_use)
def check_err_shape(pyfunc, arr, shape):
    with self.assertRaises(NotImplementedError) as raises:
        generic_run(pyfunc, arr, shape)
    self.assertEqual(str(raises.exception), 'incompatible shape for array')