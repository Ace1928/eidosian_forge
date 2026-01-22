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
def test_linear_lm_direct(reset_randomstate):
    endog = np.random.standard_normal(500)
    exog = add_constant(np.random.standard_normal((500, 3)))
    res = OLS(endog, exog).fit()
    lm_res = smsdia.linear_lm(res.resid, exog)
    aug = np.hstack([exog, exog[:, 1:] ** 2])
    res_aug = OLS(res.resid, aug).fit()
    stat = res_aug.rsquared * aug.shape[0]
    assert_allclose(lm_res[0], stat)