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
@pytest.mark.parametrize('attribute', attributes)
def test_equiv_ols_autoreg(ols_autoreg_result, attribute):
    a, o = ols_autoreg_result
    ols_a = getattr(o, attribute)
    ar_a = getattr(a, attribute)
    if callable(ols_a):
        ols_a = ols_a()
        ar_a = ar_a()
    ols_a = fix_ols_attribute(ols_a, attribute, o)
    assert_allclose(ols_a, ar_a)