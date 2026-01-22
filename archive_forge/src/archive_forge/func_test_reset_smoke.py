import json
import os
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
@pytest.mark.smoke
@pytest.mark.parametrize('power', [2, 3])
@pytest.mark.parametrize('test_type', ['fitted', 'exog', 'princomp'])
@pytest.mark.parametrize('use_f', [True, False])
@pytest.mark.parametrize('cov', [dict(cov_type='nonrobust', cov_kwargs={}), dict(cov_type='HC0', cov_kwargs={})])
def test_reset_smoke(power, test_type, use_f, cov, reset_randomstate):
    x = add_constant(np.random.standard_normal((1000, 3)))
    e = np.random.standard_normal((1000, 1))
    x = np.hstack([x, x[:, 1:] ** 2])
    y = x @ np.ones((7, 1)) + e
    res = OLS(y, x[:, :4]).fit()
    smsdia.linear_reset(res, power=power, test_type=test_type, use_f=use_f, **cov)