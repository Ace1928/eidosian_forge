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
@pytest.mark.parametrize('cov_type', ['nonrobust', 'HC0'])
def test_encompasing_direct(cov_type, reset_randomstate):
    x = np.random.standard_normal((500, 2))
    e = np.random.standard_normal((500, 1))
    x_extra = np.random.standard_normal((500, 2))
    z_extra = np.random.standard_normal((500, 3))
    y = x @ np.ones((2, 1)) + e
    x1 = np.hstack([x[:, :1], x_extra])
    z1 = np.hstack([x, z_extra])
    res1 = OLS(y, x1).fit()
    res2 = OLS(y, z1).fit()
    df = smsdia.compare_encompassing(res1, res2, cov_type=cov_type)
    direct1 = OLS(y, np.hstack([x1, x[:, 1:], z_extra])).fit(cov_type=cov_type)
    r1 = np.zeros((4, 3 + 1 + 3))
    r1[:, -4:] = np.eye(4)
    direct_test_1 = direct1.wald_test(r1, use_f=True, scalar=True)
    expected = (float(np.squeeze(direct_test_1.statistic)), float(np.squeeze(direct_test_1.pvalue)), int(direct_test_1.df_num), int(direct_test_1.df_denom))
    assert_allclose(np.asarray(df.loc['x']), expected, atol=1e-08)
    direct2 = OLS(y, np.hstack([z1, x_extra])).fit(cov_type=cov_type)
    r2 = np.zeros((2, 2 + 3 + 2))
    r2[:, -2:] = np.eye(2)
    direct_test_2 = direct2.wald_test(r2, use_f=True, scalar=True)
    expected = (float(np.squeeze(direct_test_2.statistic)), float(np.squeeze(direct_test_2.pvalue)), int(direct_test_2.df_num), int(direct_test_2.df_denom))
    assert_allclose(np.asarray(df.loc['z']), expected, atol=1e-08)