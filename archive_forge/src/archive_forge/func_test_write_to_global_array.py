import numpy as np
import unittest
from numba import njit
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import TestCase
def test_write_to_global_array(self):
    pyfunc = write_to_global_array
    with self.assertRaises(TypingError):
        njit(())(pyfunc)