import warnings
import numba
from numba import jit, njit
from numba.tests.support import TestCase, always_test
import unittest
def check_member(self, name):
    self.assertTrue(hasattr(numba, name), name)
    self.assertIn(name, numba.__all__)