import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_nrt_gen0(self):
    pygen = nrt_gen0
    cgen = jit(nopython=True)(pygen)
    py_ary = np.arange(10)
    c_ary = py_ary.copy()
    py_res = list(pygen(py_ary))
    c_res = list(cgen(c_ary))
    np.testing.assert_equal(py_ary, c_ary)
    self.assertEqual(py_res, c_res)
    self.assertRefCountEqual(py_ary, c_ary)