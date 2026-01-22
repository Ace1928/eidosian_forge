from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_squeeze_array_npm(self):
    with self.assertRaises(errors.TypingError) as raises:
        self.test_squeeze_array(flags=no_pyobj_flags)
    self.assertIn('squeeze', str(raises.exception))