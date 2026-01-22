from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_weights_intact():
    np.random.seed(12345)
    vals = np.random.lognormal(size=100)
    weights = np.random.choice([1.0, 10.0, 100], size=vals.size)
    orig_weights = weights.copy()
    stats.gaussian_kde(np.log10(vals), weights=weights)
    assert_allclose(weights, orig_weights, atol=1e-14, rtol=1e-14)