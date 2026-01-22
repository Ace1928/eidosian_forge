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
@pytest.mark.parametrize('store', [True, False])
@pytest.mark.parametrize('ddof', [0, 2])
@pytest.mark.parametrize('cov', [dict(cov_type='nonrobust', cov_kwargs={}), dict(cov_type='HC0', cov_kwargs={})])
def test_acorr_lm_smoke(store, ddof, cov, reset_randomstate):
    e = np.random.standard_normal(250)
    smsdia.acorr_lm(e, nlags=6, store=store, ddof=ddof, **cov)
    smsdia.acorr_lm(e, nlags=None, store=store, period=12, ddof=ddof, **cov)