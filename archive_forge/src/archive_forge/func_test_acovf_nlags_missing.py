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
@pytest.mark.parametrize('missing', ['conservative', 'drop'])
@pytest.mark.parametrize('fft', [False, True])
@pytest.mark.parametrize('demean', [True, False])
@pytest.mark.parametrize('adjusted', [True, False])
def test_acovf_nlags_missing(acovf_data, adjusted, demean, fft, missing):
    acovf_data = acovf_data.copy()
    acovf_data[1:3] = np.nan
    full = acovf(acovf_data, adjusted=adjusted, demean=demean, fft=fft, missing=missing)
    limited = acovf(acovf_data, adjusted=adjusted, demean=demean, fft=fft, missing=missing, nlag=10)
    assert_allclose(full[:11], limited)