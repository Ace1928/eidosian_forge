import scipy.stats
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from patsy import dmatrices  # pylint: disable=E0611
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from .results.results_quantile_regression import (
def test_collinear_matrix():
    X = np.array([[1, 0, 0.5], [1, 0, 0.8], [1, 0, 1.5], [1, 0, 0.25]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)
    res_collinear = QuantReg(y, X).fit(0.5)
    assert len(res_collinear.params) == X.shape[1]