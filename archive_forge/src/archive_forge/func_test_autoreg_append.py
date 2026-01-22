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
@pytest.mark.parametrize('trend', ['n', 'ct'])
@pytest.mark.parametrize('use_pandas', [True, False])
@pytest.mark.parametrize('lags', [0, 1, 3])
@pytest.mark.parametrize('seasonal', [True, False])
def test_autoreg_append(append_data, use_pandas, lags, trend, seasonal):
    period = 12 if not use_pandas else None
    y = append_data.y
    y_oos = append_data.y_oos
    y_both = append_data.y_both
    x = append_data.x
    x_oos = append_data.x_oos
    x_both = append_data.x_both
    if not use_pandas:
        y = np.asarray(y)
        x = np.asarray(x)
        y_oos = np.asarray(y_oos)
        x_oos = np.asarray(x_oos)
        y_both = np.asarray(y_both)
        x_both = np.asarray(x_both)
    res = AutoReg(y, lags=lags, trend=trend, seasonal=seasonal, period=period).fit()
    res_append = res.append(y_oos, refit=True)
    res_direct = AutoReg(y_both, lags=lags, trend=trend, seasonal=seasonal, period=period).fit()
    res_exog = AutoReg(y, exog=x, lags=lags, trend=trend, seasonal=seasonal, period=period).fit()
    res_exog_append = res_exog.append(y_oos, exog=x_oos, refit=True)
    res_exog_direct = AutoReg(y_both, exog=x_both, lags=lags, trend=trend, seasonal=seasonal, period=period).fit()
    assert_allclose(res_direct.params, res_append.params)
    assert_allclose(res_exog_direct.params, res_exog_append.params)
    if use_pandas:
        with pytest.raises(TypeError, match='endog must have the same type'):
            res.append(np.asarray(y_oos))
        with pytest.raises(TypeError, match='exog must have the same type'):
            res_exog.append(y_oos, np.asarray(x_oos))
    with pytest.raises(ValueError, match='Original model does'):
        res.append(y_oos, exog=x_oos)
    with pytest.raises(ValueError, match='Original model has exog'):
        res_exog.append(y_oos)