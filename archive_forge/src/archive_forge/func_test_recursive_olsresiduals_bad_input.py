import os
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises
import pandas as pd
import pytest
from scipy.stats import norm
from statsmodels.datasets import macrodata
from statsmodels.genmod.api import GLM
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.tools import add_constant
from statsmodels.tools.eval_measures import aic, bic
from statsmodels.tools.sm_exceptions import ValueWarning
def test_recursive_olsresiduals_bad_input(reset_randomstate):
    from statsmodels.tsa.arima.model import ARIMA
    e = np.random.standard_normal(250)
    y = e.copy()
    for i in range(1, y.shape[0]):
        y[i] += 0.1 + 0.8 * y[i - 1] + e[i]
    res = ARIMA(y[20:], order=(1, 0, 0), trend='c').fit()
    with pytest.raises(TypeError, match='res a regression results instance'):
        recursive_olsresiduals(res)