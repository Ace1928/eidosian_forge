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
def test_hac(self):
    res = self.res
    cov_hac_4 = np.array([1.385551290884014, -0.3133096102522685, -0.0597207976835705, -0.3133096102522685, 0.1081011690351306, 0.000389440793564336, -0.0597207976835705, 0.000389440793564339, 0.0862118527405036]).reshape((3, 3), order='F')
    cov_hac_10 = np.array([1.257386180080192, -0.2871560199899846, -0.03958300024627573, -0.2871560199899845, 0.1049107028987101, 0.0003896205316866944, -0.03958300024627578, 0.0003896205316866961, 0.0985539340694839]).reshape((3, 3), order='F')
    cov = sw.cov_hac_simple(res, nlags=4, use_correction=False)
    bse_hac = sw.se_cov(cov)
    assert_almost_equal(cov, cov_hac_4, decimal=12)
    assert_almost_equal(bse_hac, np.sqrt(np.diag(cov)), decimal=12)
    cov = sw.cov_hac_simple(res, nlags=10, use_correction=False)
    bse_hac = sw.se_cov(cov)
    assert_almost_equal(cov, cov_hac_10, decimal=12)
    assert_almost_equal(bse_hac, np.sqrt(np.diag(cov)), decimal=12)