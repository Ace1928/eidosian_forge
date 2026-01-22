import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_random_float_scalar(self):
    random = Generator(MT19937(self.seed))
    actual = random.random(dtype=np.float32)
    desired = 0.0969992
    assert_array_almost_equal(actual, desired, decimal=7)