import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
@TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': ''})
def test_boundscheck_unset(self):
    self.assertIsNone(config.BOUNDSCHECK)
    a = np.array([1])
    self.default(a)
    self.off(a)
    with self.assertRaises(IndexError):
        self.on(a)