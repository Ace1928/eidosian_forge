import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.slow
@pytest.mark.parametrize('sample_size,high,dtype,chi2max', [(5000000, 5, np.int8, 125.0), (5000000, 7, np.uint8, 150.0), (10000000, 2500, np.int16, 3300.0), (50000000, 5000, np.uint16, 6500.0)])
def test_integers_small_dtype_chisquared(self, sample_size, high, dtype, chi2max):
    samples = random.integers(high, size=sample_size, dtype=dtype)
    values, counts = np.unique(samples, return_counts=True)
    expected = sample_size / high
    chi2 = ((counts - expected) ** 2 / expected).sum()
    assert chi2 < chi2max