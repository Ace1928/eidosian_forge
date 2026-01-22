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
def test_encompasing_error(reset_randomstate):
    x = np.random.standard_normal((500, 2))
    e = np.random.standard_normal((500, 1))
    z_extra = np.random.standard_normal((500, 3))
    y = x @ np.ones((2, 1)) + e
    z = np.hstack([x, z_extra])
    res1 = OLS(y, x).fit()
    res2 = OLS(y, z).fit()
    with pytest.raises(ValueError, match='The exog in results_x'):
        smsdia.compare_encompassing(res1, res2)
    with pytest.raises(TypeError, match='results_z must come from a linear'):
        smsdia.compare_encompassing(res1, 2)
    with pytest.raises(TypeError, match='results_x must come from a linear'):
        smsdia.compare_encompassing(4, 2)