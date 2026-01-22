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
def test_autoreg_apply_exception(reset_randomstate):
    y = np.random.standard_normal(250)
    mod = AutoReg(y, lags=10)
    res = mod.fit()
    with pytest.raises(ValueError, match='An exception occured'):
        res.apply(y[:5])
    x = np.random.standard_normal((y.shape[0], 3))
    res = AutoReg(y, lags=1, exog=x).fit()
    with pytest.raises(ValueError, match='exog must be provided'):
        res.apply(y[50:150])
    x = np.random.standard_normal((y.shape[0], 3))
    res = AutoReg(y, lags=1, exog=x).fit()
    with pytest.raises(ValueError, match='The number of exog'):
        res.apply(y[50:150], exog=x[50:150, :2])
    res = AutoReg(y, lags=1).fit()
    with pytest.raises(ValueError, match='exog must be None'):
        res.apply(y[50:150], exog=x[50:150])