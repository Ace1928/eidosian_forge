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
def test_innovations_filter_errors():
    ma = -0.9
    acovf = np.array([1 + ma ** 2, ma])
    theta, _ = innovations_algo(acovf, nobs=4)
    with pytest.raises(ValueError):
        innovations_filter(np.empty((2, 2)), theta)
    with pytest.raises(ValueError):
        innovations_filter(np.empty(4), theta[:-1])
    with pytest.raises(ValueError):
        innovations_filter(pd.DataFrame(np.empty((1, 4))), theta)