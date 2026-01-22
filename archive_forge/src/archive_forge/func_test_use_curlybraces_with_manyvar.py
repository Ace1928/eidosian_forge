import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_use_curlybraces_with_manyvar(self):

    @njit
    def foo(x, y):
        return {x: 1, y: x + y}
    x, y = (10, 20)
    self.assertEqual(foo(x, y), foo.py_func(x, y))