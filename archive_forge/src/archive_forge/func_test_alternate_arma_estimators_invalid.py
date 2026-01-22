import numpy as np
import pytest
from numpy.testing import (
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake, oshorts
from statsmodels.tsa.arima.estimators.gls import gls
def test_alternate_arma_estimators_invalid():
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 1), arma_estimator='invalid_estimator')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 1), arma_estimator='yule_walker')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 0), seasonal_order=(1, 0, 0, 4), arma_estimator='yule_walker')
    assert_raises(ValueError, gls, endog, exog, order=([0, 1], 0, 0), arma_estimator='yule_walker')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 1), arma_estimator='burg')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 0), seasonal_order=(1, 0, 0, 4), arma_estimator='burg')
    assert_raises(ValueError, gls, endog, exog, order=([0, 1], 0, 0), arma_estimator='burg')
    assert_raises(ValueError, gls, endog, exog, order=(1, 0, 0), arma_estimator='innovations')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 0), seasonal_order=(0, 0, 1, 4), arma_estimator='innovations')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, [0, 1]), arma_estimator='innovations')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 0), seasonal_order=(0, 0, 1, 4), arma_estimator='hannan_rissanen')