import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_curlybraces_init_with_coercion(self):

    @njit
    def foo():
        return {1: 2.2, 3: 4, 5: 6}
    self.assertEqual(foo(), foo.py_func())