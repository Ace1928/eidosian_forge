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
def test_ljungbox_auto_lag_selection(reset_randomstate):
    data = sunspots.load_pandas().data['SUNACTIVITY']
    res = AutoReg(data, 4, old_names=False).fit()
    resid = res.resid
    res1 = smsdia.acorr_ljungbox(resid, auto_lag=True)
    res2 = smsdia.acorr_ljungbox(resid, model_df=4, auto_lag=True)
    assert_allclose(res1.iloc[:, 0], res2.iloc[:, 0])
    assert res1.shape[0] >= 1
    assert res2.shape[0] >= 1
    assert np.all(np.isnan(res2.iloc[:4, 1]))
    assert np.all(res2.iloc[4:, 1] <= res1.iloc[4:, 1])