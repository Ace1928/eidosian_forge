import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def test_global_record(self):
    x = np.recarray(1, dtype=x_dt)[0]
    x.a = 1
    res = global_record_func(x)
    self.assertEqual(True, res)
    x.a = 2
    res = global_record_func(x)
    self.assertEqual(False, res)