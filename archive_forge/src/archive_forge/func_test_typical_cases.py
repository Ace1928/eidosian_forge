import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('nsample', [8, 25, 45, 55])
@pytest.mark.parametrize('method', ['count', 'marginals'])
@pytest.mark.parametrize('size', [5, (2, 3), 150000])
def test_typical_cases(self, nsample, method, size):
    random = Generator(MT19937(self.seed))
    colors = np.array([10, 5, 20, 25])
    sample = random.multivariate_hypergeometric(colors, nsample, size, method=method)
    if isinstance(size, int):
        expected_shape = (size,) + colors.shape
    else:
        expected_shape = size + colors.shape
    assert_equal(sample.shape, expected_shape)
    assert_((sample >= 0).all())
    assert_((sample <= colors).all())
    assert_array_equal(sample.sum(axis=-1), np.full(size, fill_value=nsample, dtype=int))
    if isinstance(size, int) and size >= 100000:
        assert_allclose(sample.mean(axis=0), nsample * colors / colors.sum(), rtol=0.001, atol=0.005)