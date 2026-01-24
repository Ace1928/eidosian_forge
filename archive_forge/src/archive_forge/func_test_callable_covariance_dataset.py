from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_callable_covariance_dataset(self):
    """Test the callable's cov factor for a multi-dimensional array."""
    np.random.seed(8928678)
    n_basesample = 50
    multidim_data = [np.random.randn(n_basesample) for i in range(5)]

    def callable_fun(x):
        return 0.55
    kde = mlab.GaussianKDE(multidim_data, bw_method=callable_fun)
    assert kde.covariance_factor() == 0.55