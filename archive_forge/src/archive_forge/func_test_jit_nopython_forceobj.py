import warnings
import numba
from numba import jit, njit
from numba.tests.support import TestCase, always_test
import unittest
def test_jit_nopython_forceobj(self):
    with self.assertRaises(ValueError) as cm:
        jit(nopython=True, forceobj=True)
    self.assertIn("Only one of 'nopython' or 'forceobj' can be True.", str(cm.exception))

    def py_func(x):
        return x
    jit_func = jit(nopython=True)(py_func)
    jit_func(1)
    self.assertEqual(len(jit_func.nopython_signatures), 1)
    jit_func = jit(forceobj=True)(py_func)
    jit_func(1)
    self.assertEqual(len(jit_func.nopython_signatures), 0)