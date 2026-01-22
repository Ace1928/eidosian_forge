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
def test_outlier_influence_funcs(reset_randomstate):
    x = add_constant(np.random.randn(10, 2))
    y = x.sum(1) + np.random.randn(10)
    res = OLS(y, x).fit()
    out_05 = oi.summary_table(res)
    out_01 = oi.summary_table(res, alpha=0.01)
    assert_(np.all(out_01[1][:, 6] <= out_05[1][:, 6]))
    assert_(np.all(out_01[1][:, 7] >= out_05[1][:, 7]))
    res2 = OLS(y, x[:, 0]).fit()
    oi.summary_table(res2, alpha=0.05)
    infl = res2.get_influence()
    infl.summary_table()