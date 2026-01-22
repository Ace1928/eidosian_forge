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
def test_equiv_dynamic(reset_randomstate):
    e = np.random.standard_normal(1001)
    y = np.empty(1001)
    y[0] = e[0] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    for i in range(1, 1001):
        y[i] = 0.9 * y[i - 1] + e[i]
    mod = AutoReg(y, 1)
    res = mod.fit()
    pred0 = res.predict(500, 800, dynamic=0)
    pred1 = res.predict(500, 800, dynamic=True)
    idx = pd.date_range(dt.datetime(2000, 1, 30), periods=1001, freq=MONTH_END)
    y = pd.Series(y, index=idx)
    mod = AutoReg(y, 1)
    res = mod.fit()
    pred2 = res.predict(idx[500], idx[800], dynamic=idx[500])
    pred3 = res.predict(idx[500], idx[800], dynamic=0)
    pred4 = res.predict(idx[500], idx[800], dynamic=True)
    assert_allclose(pred0, pred1)
    assert_allclose(pred0, pred2)
    assert_allclose(pred0, pred3)
    assert_allclose(pred0, pred4)