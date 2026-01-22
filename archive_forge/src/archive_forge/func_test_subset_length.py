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
@pytest.mark.parametrize('subset_length,expected_statistic,expected_pvalue', [(2, 41, 2 * min(f.cdf(41, 2, 2), f.sf(41, 2, 2))), (0.5, 10, 2 * min(f.cdf(10, 3, 3), f.sf(10, 3, 3)))])
def test_subset_length(self, subset_length, expected_statistic, expected_pvalue):
    input_residuals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    actual_statistic, actual_pvalue = breakvar_heteroskedasticity_test(input_residuals, subset_length=subset_length)
    assert actual_statistic == expected_statistic
    assert actual_pvalue == expected_pvalue