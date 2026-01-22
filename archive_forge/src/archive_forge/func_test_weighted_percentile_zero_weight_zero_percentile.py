import numpy as np
from numpy.testing import assert_allclose
from pytest import approx
from sklearn.utils.stats import _weighted_percentile
def test_weighted_percentile_zero_weight_zero_percentile():
    y = np.array([0, 1, 2, 3, 4, 5])
    sw = np.array([0, 0, 1, 1, 1, 0])
    score = _weighted_percentile(y, sw, 0)
    assert approx(score) == 2
    score = _weighted_percentile(y, sw, 50)
    assert approx(score) == 3
    score = _weighted_percentile(y, sw, 100)
    assert approx(score) == 4