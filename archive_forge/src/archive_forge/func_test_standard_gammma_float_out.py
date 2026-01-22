import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_standard_gammma_float_out(self):
    actual = np.zeros((3, 2), dtype=np.float32)
    random = Generator(MT19937(self.seed))
    random.standard_gamma(10.0, out=actual, dtype=np.float32)
    desired = np.array([[10.14987, 7.87012], [9.46284, 12.56832], [13.82495, 7.81533]], dtype=np.float32)
    assert_array_almost_equal(actual, desired, decimal=5)
    random = Generator(MT19937(self.seed))
    random.standard_gamma(10.0, out=actual, size=(3, 2), dtype=np.float32)
    assert_array_almost_equal(actual, desired, decimal=5)