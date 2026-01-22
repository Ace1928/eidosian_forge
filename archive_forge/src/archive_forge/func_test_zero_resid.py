import scipy.stats
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from patsy import dmatrices  # pylint: disable=E0611
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from .results.results_quantile_regression import (
def test_zero_resid():
    X = np.array([[1, 0], [0, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)
    res = QuantReg(y, X).fit(0.5, bandwidth='chamberlain')
    res.summary()
    assert_allclose(res.params, np.array([0.0, 0.96774163]), rtol=0.0001, atol=1e-20)
    assert_allclose(res.bse, np.array([0.0447576, 0.01154867]), rtol=0.0001, atol=1e-20)
    assert_allclose(res.resid, np.array([0.0, 0.032258368, -0.0322574272, 9.40732912e-07]), rtol=0.0001, atol=1e-20)
    X = np.array([[1, 0], [0.1, 1], [0, 2.1], [0, 3.1]], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)
    res = QuantReg(y, X).fit(0.5, bandwidth='chamberlain')
    res.summary()
    assert_allclose(res.params, np.array([9.99982796e-08, 0.96774163]), rtol=0.0001, atol=1e-20)
    assert_allclose(res.bse, np.array([0.04455029, 0.01155251]), rtol=0.0001, atol=1e-20)
    assert_allclose(res.resid, np.array([-9.99982796e-08, 0.0322583598, -0.0322574234, 9.4636186e-07]), rtol=0.0001, atol=1e-20)