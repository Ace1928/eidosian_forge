import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_unituple_index_error(self):

    def pyfunc(a, i):
        return a.shape[i]
    cfunc = njit((types.Array(types.int32, 1, 'A'), types.int32))(pyfunc)
    a = np.empty(2, dtype=np.int32)
    self.assertEqual(cfunc(a, 0), pyfunc(a, 0))
    with self.assertRaises(IndexError) as cm:
        cfunc(a, 2)
    self.assertEqual(str(cm.exception), 'tuple index out of range')