from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_flatten_array_npm(self):
    self.test_flatten_array(flags=no_pyobj_flags)
    self.test_flatten_array(flags=no_pyobj_flags, layout='F')
    self.test_flatten_array(flags=no_pyobj_flags, layout='A')