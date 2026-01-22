from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import datetime as dt
from itertools import product
from typing import NamedTuple, Union
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
from pandas import Index, Series, date_range, period_range
from pandas.testing import assert_series_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import SpecificationWarning, ValueWarning
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tests.results import results_ar
def test_autoreg_predict_forecast_equiv(reset_randomstate):
    e = np.random.normal(size=1000)
    nobs = e.shape[0]
    idx = pd.date_range(dt.datetime(2020, 1, 1), freq='D', periods=nobs)
    for i in range(1, nobs):
        e[i] = 0.95 * e[i - 1] + e[i]
    y = pd.Series(e, index=idx)
    m = AutoReg(y, trend='c', lags=1)
    res = m.fit()
    a = res.forecast(12)
    b = res.predict(nobs, nobs + 11)
    c = res.forecast('2022-10-08')
    assert_series_equal(a, b)
    assert_series_equal(a, c)
    sarimax_res = SARIMAX(y, order=(1, 0, 0), trend='c').fit(disp=False)
    d = sarimax_res.forecast(12)
    pd.testing.assert_index_equal(a.index, d.index)