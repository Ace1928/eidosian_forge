from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_evaluate_inv_dim(self):
    """
        Invert the dimensions; i.e., for a dataset of dimension 1 [3, 2, 4],
        the points should have a dimension of 3 [[3], [2], [4]].
        """
    np.random.seed(8928678)
    n_basesample = 50
    multidim_data = np.random.randn(n_basesample)
    kde = mlab.GaussianKDE(multidim_data)
    x2 = [[1], [2], [3]]
    with pytest.raises(ValueError):
        kde.evaluate(x2)