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
def fix_ols_attribute(val, attrib, res):
    """
    fixes to correct for df adjustment b/t OLS and AutoReg with nonrobust cov
    """
    nparam = res.k_constant + res.df_model
    nobs = nparam + res.df_resid
    df_correction = (nobs - nparam) / nobs
    if attrib in ('scale',):
        return val * df_correction
    elif attrib == 'df_model':
        return val + res.k_constant
    elif res.cov_type != 'nonrobust':
        return val
    elif attrib in ('bse', 'conf_int'):
        return val * np.sqrt(df_correction)
    elif attrib in ('cov_params', 'scale'):
        return val * df_correction
    elif attrib in ('f_test',):
        return val / df_correction
    elif attrib in ('tvalues',):
        return val / np.sqrt(df_correction)
    return val