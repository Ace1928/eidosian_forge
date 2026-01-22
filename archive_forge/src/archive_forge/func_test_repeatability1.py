import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_repeatability1(self):
    random = Generator(MT19937(self.seed))
    sample = random.multivariate_hypergeometric([3, 4, 5], 5, size=5, method='count')
    expected = np.array([[2, 1, 2], [2, 1, 2], [1, 1, 3], [2, 0, 3], [2, 1, 2]])
    assert_array_equal(sample, expected)