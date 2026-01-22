import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
@TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': ''})
def test_basic_array_boundscheck(self):
    self.assertIsNone(config.BOUNDSCHECK)
    a = np.arange(5)
    with self.assertRaises(IndexError):
        basic_array_access(a)
    at = typeof(a)
    boundscheck = njit((at,), boundscheck=True)(basic_array_access)
    with self.assertRaises(IndexError):
        boundscheck(a)