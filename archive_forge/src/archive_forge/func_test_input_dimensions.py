from statsmodels.compat.python import lrange, lmap
import os
import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm
def test_input_dimensions(self):
    rs = np.random.RandomState(1234)
    x = rs.randn(200, 2)
    z = rs.randn(200)
    x[:, 0] = np.sqrt(0.5) * x[:, 0] + np.sqrt(0.5) * z
    z = np.column_stack((x[:, [1]], z[:, None]))
    e = np.sqrt(0.5) * rs.randn(200) + np.sqrt(0.5) * x[:, 0]
    y_1d = y = x[:, 0] + x[:, 1] + e
    y_2d = y[:, None]
    y_series = pd.Series(y)
    y_df = pd.DataFrame(y_series)
    x_1d = x[:, 0]
    x_2d = x
    x_df = pd.DataFrame(x)
    x_df_single = x_df.iloc[:, [0]]
    x_series = x_df.iloc[:, 0]
    z_2d = z
    z_series = pd.Series(z[:, 1])
    z_1d = z_series.values
    z_df = pd.DataFrame(z)
    ys = (y_df, y_series, y_2d, y_1d)
    xs = (x_2d, x_1d, x_df_single, x_df, x_series)
    zs = (z_1d, z_2d, z_series, z_df)
    res2 = gmm.IV2SLS(y_1d, x_2d, z_2d).fit()
    res1 = gmm.IV2SLS(y_1d, x_1d, z_1d).fit()
    res1_2sintr = gmm.IV2SLS(y_1d, x_1d, z_2d).fit()
    for _y in ys:
        for _x in xs:
            for _z in zs:
                x_1d = np.size(_x) == _x.shape[0]
                z_1d = np.size(_z) == _z.shape[0]
                if z_1d and (not x_1d):
                    continue
                res = gmm.IV2SLS(_y, _x, _z).fit()
                if z_1d:
                    assert_allclose(res.params, res1.params)
                elif x_1d and (not z_1d):
                    assert_allclose(res.params, res1_2sintr.params)
                else:
                    assert_allclose(res.params, res2.params)