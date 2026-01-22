import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_a_is_not_b_intp(self):
    pyfunc = a_is_not_b
    cfunc = njit((types.intp, types.intp))(pyfunc)
    self.assertFalse(cfunc(1, 1))
    self.assertTrue(cfunc(1, 2))