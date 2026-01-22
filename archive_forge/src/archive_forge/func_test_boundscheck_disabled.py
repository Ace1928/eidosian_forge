import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
@TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': '0'})
def test_boundscheck_disabled(self):
    self.assertFalse(config.BOUNDSCHECK)
    a = np.array([1])
    self.default(a)
    self.off(a)
    self.on(a)