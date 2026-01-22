import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
@TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': '1'})
def test_boundscheck_enabled(self):
    self.assertTrue(config.BOUNDSCHECK)
    a = np.array([1])
    with self.assertRaises(IndexError):
        self.default(a)
        self.off(a)
        self.on(a)