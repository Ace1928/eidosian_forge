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
@pytest.mark.matplotlib
def test_parameterless_autoreg():
    data = gen_data(250, 0, False)
    mod = AutoReg(data.endog, 0, trend='n', seasonal=False, exog=None)
    res = mod.fit()
    for attr in dir(res):
        if attr.startswith('_'):
            continue
        if attr in ('predict', 'f_test', 't_test', 'initialize', 'load', 'remove_data', 'save', 't_test', 't_test_pairwise', 'wald_test', 'wald_test_terms', 'apply', 'append'):
            continue
        attr = getattr(res, attr)
        if callable(attr):
            attr()
        else:
            assert isinstance(attr, object)