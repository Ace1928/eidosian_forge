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
def test_autoreg_append_deterministic(append_data):
    y = append_data.y
    y_oos = append_data.y_oos
    y_both = append_data.y_both
    x = append_data.x
    x_oos = append_data.x_oos
    x_both = append_data.x_both
    terms = [TimeTrend(constant=True, order=1), Seasonality(12)]
    dp = DeterministicProcess(y.index, additional_terms=terms)
    res = AutoReg(y, lags=3, trend='n', deterministic=dp).fit()
    res_append = res.append(y_oos, refit=True)
    res_direct = AutoReg(y_both, lags=3, trend='n', deterministic=dp.apply(y_both.index)).fit()
    assert_allclose(res_append.params, res_direct.params)
    res_np = AutoReg(np.asarray(y), lags=3, trend='n', deterministic=dp).fit()
    res_append_np = res_np.append(np.asarray(y_oos))
    assert_allclose(res_np.params, res_append_np.params)
    res = AutoReg(y, exog=x, lags=3, trend='n', deterministic=dp).fit()
    res_append = res.append(y_oos, exog=x_oos, refit=True)
    res_direct = AutoReg(y_both, exog=x_both, lags=3, trend='n', deterministic=dp.apply(y_both.index)).fit()
    assert_allclose(res_append.params, res_direct.params)