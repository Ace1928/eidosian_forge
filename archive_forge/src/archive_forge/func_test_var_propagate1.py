import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_var_propagate1(self):
    cfunc = njit((types.intp, types.intp))(var_propagate1)
    self.run_propagate_func(cfunc, (2, 3))
    self.run_propagate_func(cfunc, (3, 2))