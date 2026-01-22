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
def test_autoreg_constant_column_trend():
    sample = np.array([0.46341460943222046, 0.46341460943222046, 0.39024388790130615, 0.4146341383457184, 0.4146341383457184, 0.4146341383457184, 0.3414634168148041, 0.4390243887901306, 0.46341460943222046, 0.4390243887901306])
    with pytest.raises(ValueError, match='The model specification cannot'):
        AutoReg(sample, lags=7)
    with pytest.raises(ValueError, match='The model specification cannot'):
        AutoReg(sample, lags=7, trend='n')