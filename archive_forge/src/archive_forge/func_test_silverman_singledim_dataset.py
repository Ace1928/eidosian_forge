from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_silverman_singledim_dataset(self):
    """Test silverman's output for a single dimension list."""
    x1 = np.array([-7, -5, 1, 4, 5])
    mygauss = mlab.GaussianKDE(x1, 'silverman')
    y_expected = 0.767703899274755
    assert_almost_equal(mygauss.covariance_factor(), y_expected, 7)