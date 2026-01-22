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
@pytest.fixture(scope='module')
def diagnostic_data():
    rs = np.random.RandomState(93674328)
    e = rs.standard_normal(500)
    x = rs.standard_normal((500, 3))
    y = x.sum(1) + e
    c = np.ones_like(y)
    data = pd.DataFrame(np.c_[y, c, x], columns=['y', 'c', 'x1', 'x2', 'x3'])
    return data