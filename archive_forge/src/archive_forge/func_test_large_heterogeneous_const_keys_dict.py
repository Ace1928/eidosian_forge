import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_large_heterogeneous_const_keys_dict(self):
    """
        Tests that a function with a large heterogeneous constant
        dictionary remains a constant.
        """

    def const_keys_func(a):
        d = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': a}
        return d['S']
    py_func = const_keys_func
    cfunc = njit()(const_keys_func)
    value = 'a_string'
    a = py_func(value)
    b = cfunc(value)
    self.assertEqual(a, b)