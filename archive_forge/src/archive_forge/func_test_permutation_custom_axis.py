import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_permutation_custom_axis(self):
    a = np.arange(16).reshape((4, 4))
    desired = np.array([[0, 3, 1, 2], [4, 7, 5, 6], [8, 11, 9, 10], [12, 15, 13, 14]])
    random = Generator(MT19937(self.seed))
    actual = random.permutation(a, axis=1)
    assert_array_equal(actual, desired)
    random = Generator(MT19937(self.seed))
    actual = random.permutation(a, axis=-1)
    assert_array_equal(actual, desired)