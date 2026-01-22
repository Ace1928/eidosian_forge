from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_scalar_covariance_dataset(self):
    """Test a scalar's cov factor."""
    np.random.seed(8765678)
    n_basesample = 50
    multidim_data = [np.random.randn(n_basesample) for i in range(5)]
    kde = mlab.GaussianKDE(multidim_data, bw_method=0.5)
    assert kde.covariance_factor() == 0.5