import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_repeatability3(self):
    random = Generator(MT19937(self.seed))
    sample = random.multivariate_hypergeometric([20, 30, 50], 12, size=5, method='marginals')
    expected = np.array([[2, 3, 7], [5, 3, 4], [2, 5, 5], [5, 3, 4], [1, 5, 6]])
    assert_array_equal(sample, expected)