from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_diagnostics_nile_eviews():
    niledata = nile.data.load_pandas().data
    niledata.index = pd.date_range('1871-01-01', '1970-01-01', freq='YS')
    mod = MLEModel(niledata['volume'], k_states=1, initialization='approximate_diffuse', initial_variance=1000000000000000.0, loglikelihood_burn=1)
    mod.ssm['design', 0, 0] = 1
    mod.ssm['obs_cov', 0, 0] = np.exp(9.60035)
    mod.ssm['transition', 0, 0] = 1
    mod.ssm['selection', 0, 0] = 1
    mod.ssm['state_cov', 0, 0] = np.exp(7.348705)
    res = mod.filter([])
    actual = res.test_serial_correlation(method='ljungbox', lags=10)[0, :, -1]
    assert_allclose(actual, [13.117, 0.217], atol=0.001)
    actual = res.test_normality(method='jarquebera')[0, :2]
    assert_allclose(actual, [0.041686, 0.979373], atol=1e-05)