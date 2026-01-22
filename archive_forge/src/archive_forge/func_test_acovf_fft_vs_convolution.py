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
@pytest.mark.parametrize('demean', [True, False])
@pytest.mark.parametrize('adjusted', [True, False])
def test_acovf_fft_vs_convolution(demean, adjusted, reset_randomstate):
    q = np.random.normal(size=100)
    F1 = acovf(q, demean=demean, adjusted=adjusted, fft=True)
    F2 = acovf(q, demean=demean, adjusted=adjusted, fft=False)
    assert_almost_equal(F1, F2, decimal=7)