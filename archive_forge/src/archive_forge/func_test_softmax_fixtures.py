import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from scipy.special import logsumexp, softmax
def test_softmax_fixtures():
    assert_allclose(softmax([1000, 0, 0, 0]), np.array([1, 0, 0, 0]), rtol=1e-13)
    assert_allclose(softmax([1, 1]), np.array([0.5, 0.5]), rtol=1e-13)
    assert_allclose(softmax([0, 1]), np.array([1, np.e]) / (1 + np.e), rtol=1e-13)
    x = np.arange(4)
    expected = np.array([0.03205860328008499, 0.08714431874203256, 0.23688281808991013, 0.6439142598879722])
    assert_allclose(softmax(x), expected, rtol=1e-13)
    assert_allclose(softmax(x + 100), expected, rtol=1e-13)
    assert_allclose(softmax(x.reshape(2, 2)), expected.reshape(2, 2), rtol=1e-13)