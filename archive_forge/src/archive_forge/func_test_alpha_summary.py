import scipy.stats
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from patsy import dmatrices  # pylint: disable=E0611
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from .results.results_quantile_regression import (
def test_alpha_summary():
    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)
    res = QuantReg(y, X).fit(0.5, bandwidth='chamberlain', use_t=True)
    summ_20 = res.summary(alpha=0.2)
    assert '[0.025      0.975]' not in str(summ_20)
    assert '[0.1        0.9]' in str(summ_20)