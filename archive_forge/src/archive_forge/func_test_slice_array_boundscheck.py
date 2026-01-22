import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
@TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': ''})
def test_slice_array_boundscheck(self):
    self.assertIsNone(config.BOUNDSCHECK)
    a = np.ones((5, 5))
    b = np.ones((5, 20))
    with self.assertRaises(IndexError):
        slice_array_access(a)
    slice_array_access(b)
    at = typeof(a)
    rt = float64[:]
    boundscheck = njit(rt(at), boundscheck=True)(slice_array_access)
    with self.assertRaises(IndexError):
        boundscheck(a)