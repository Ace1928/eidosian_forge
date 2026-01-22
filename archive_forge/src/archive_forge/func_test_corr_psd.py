import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
def test_corr_psd():
    x = np.array([[1, -0.2, -0.9], [-0.2, 1, -0.2], [-0.9, -0.2, 1]])
    y = corr_nearest(x, n_fact=100)
    assert_almost_equal(x, y, decimal=14)
    y = corr_clipped(x)
    assert_almost_equal(x, y, decimal=14)
    y = cov_nearest(x, n_fact=100)
    assert_almost_equal(x, y, decimal=14)
    x2 = x + 0.001 * np.eye(3)
    y = cov_nearest(x2, n_fact=100)
    assert_almost_equal(x2, y, decimal=14)