import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_integers_masked(self):
    random = Generator(MT19937(self.seed))
    actual = random.integers(0, 99, size=(3, 2), dtype=np.uint32)
    desired = np.array([[9, 21], [70, 68], [8, 41]], dtype=np.uint32)
    assert_array_equal(actual, desired)