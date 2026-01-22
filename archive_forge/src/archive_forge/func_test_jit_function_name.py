import unittest
from numba import jit
def test_jit_function_name(self):

    def add(x, y):
        return x + y
    c_add = jit(add)
    self.assertEqual(c_add.__name__, 'add')