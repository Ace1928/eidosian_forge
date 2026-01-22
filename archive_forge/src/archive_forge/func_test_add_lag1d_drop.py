from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
def test_add_lag1d_drop(self):
    data = self.random_data
    lagmat = stattools.lagmat(data, 3, trim='Both')
    lag_data = tools.add_lag(data, lags=3, drop=True, insert=True)
    assert_equal(lagmat, lag_data)
    lag_data = tools.add_lag(data, lags=3, drop=True, insert=False)
    assert_equal(lagmat, lag_data)