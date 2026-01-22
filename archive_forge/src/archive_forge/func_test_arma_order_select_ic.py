from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import MONTH_END, YEAR_END, assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import lrange
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
@pytest.mark.smoke
@pytest.mark.slow
def test_arma_order_select_ic():
    from statsmodels.tsa.arima_process import arma_generate_sample
    arparams = np.array([0.75, -0.25])
    maparams = np.array([0.65, 0.35])
    arparams = np.r_[1, -arparams]
    maparam = np.r_[1, maparams]
    nobs = 250
    np.random.seed(2014)
    y = arma_generate_sample(arparams, maparams, nobs)
    res = arma_order_select_ic(y, ic=['aic', 'bic'], trend='n')
    aic_x = np.array([[764.36517643, 552.7342255, 484.29687843], [562.10924262, 485.5197969, 480.32858497], [507.04581344, 482.91065829, 481.91926034], [484.03995962, 482.14868032, 483.86378955], [481.8849479, 483.8377379, 485.83756612]])
    bic_x = np.array([[767.88663735, 559.77714733, 494.86126118], [569.15216446, 496.08417966, 494.41442864], [517.61019619, 496.99650196, 499.52656493], [498.12580329, 499.75598491, 504.99255506], [499.49225249, 504.96650341, 510.48779255]])
    aic = DataFrame(aic_x, index=lrange(5), columns=lrange(3))
    bic = DataFrame(bic_x, index=lrange(5), columns=lrange(3))
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert_almost_equal(res.bic.values, bic.values, 5)
    assert_equal(res.aic_min_order, (1, 2))
    assert_equal(res.bic_min_order, (1, 2))
    assert_(res.aic.index.equals(aic.index))
    assert_(res.aic.columns.equals(aic.columns))
    assert_(res.bic.index.equals(bic.index))
    assert_(res.bic.columns.equals(bic.columns))
    index = pd.date_range('2000-1-1', freq=MONTH_END, periods=len(y))
    y_series = pd.Series(y, index=index)
    res_pd = arma_order_select_ic(y_series, max_ar=2, max_ma=1, ic=['aic', 'bic'], trend='n')
    assert_almost_equal(res_pd.aic.values, aic.values[:3, :2], 5)
    assert_almost_equal(res_pd.bic.values, bic.values[:3, :2], 5)
    assert_equal(res_pd.aic_min_order, (2, 1))
    assert_equal(res_pd.bic_min_order, (1, 1))
    res = arma_order_select_ic(y, ic='aic', trend='n')
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert_(res.aic.index.equals(aic.index))
    assert_(res.aic.columns.equals(aic.columns))
    assert_equal(res.aic_min_order, (1, 2))