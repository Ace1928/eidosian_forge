import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_var_propagate4(self):
    cfunc = njit((types.intp, types.intp))(var_propagate4)
    self.run_propagate_func(cfunc, (1, 1))
    self.run_propagate_func(cfunc, (1, 0))
    self.run_propagate_func(cfunc, (1, -1))
    self.run_propagate_func(cfunc, (0, 1))
    self.run_propagate_func(cfunc, (0, 0))
    self.run_propagate_func(cfunc, (0, -1))
    self.run_propagate_func(cfunc, (-1, 1))
    self.run_propagate_func(cfunc, (-1, 0))
    self.run_propagate_func(cfunc, (-1, -1))