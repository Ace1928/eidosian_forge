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
def test_predict_irregular_ar():
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    y[:3] = e[:3] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    for i in range(3, 1001):
        y[i] = 10 + 0.9 * y[i - 1] - 0.5 * y[i - 3] + e[i]
    ys = pd.Series(y, index=pd.date_range(dt.datetime(1950, 1, 1), periods=1001, freq=MONTH_END))
    mod = AutoReg(ys, [1, 3], trend='ct')
    res = mod.fit()
    c = res.params.iloc[0]
    t = res.params.iloc[1]
    ar = np.asarray(res.params.iloc[2:])
    pred = res.predict(900, 1100, True)
    direct = np.zeros(201)
    direct[0] = c + t * 901 + ar[0] * y[899] + ar[1] * y[897]
    direct[1] = c + t * 902 + ar[0] * direct[0] + ar[1] * y[898]
    direct[2] = c + t * 903 + ar[0] * direct[1] + ar[1] * y[899]
    for i in range(3, 201):
        direct[i] = c + t * (901 + i) + ar[0] * direct[i - 1] + ar[1] * direct[i - 3]
    direct = pd.Series(direct, index=pd.date_range(ys.index[900], periods=201, freq=MONTH_END))
    assert_series_equal(pred, direct)
    pred = res.predict(900)
    direct = c + t * np.arange(901, 901 + 101) + ar[0] * y[899:-1] + ar[1] * y[897:-3]
    idx = pd.date_range(ys.index[900], periods=101, freq=MONTH_END)
    direct = pd.Series(direct, index=idx)
    assert_series_equal(pred, direct)