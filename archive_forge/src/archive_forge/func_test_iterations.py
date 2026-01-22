import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake, oshorts
from statsmodels.tsa.arima.estimators.gls import gls
def test_iterations():
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    _, res = gls(endog, exog, order=(1, 0, 0), n_iter=1)
    assert_equal(res.iterations, 1)
    assert_equal(res.converged, None)