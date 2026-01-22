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
def test_diagnostics_nile_durbinkoopman():
    niledata = nile.data.load_pandas().data
    niledata.index = pd.date_range('1871-01-01', '1970-01-01', freq='YS')
    mod = MLEModel(niledata['volume'], k_states=1, initialization='approximate_diffuse', initial_variance=1000000000000000.0, loglikelihood_burn=1)
    mod.ssm['design', 0, 0] = 1
    mod.ssm['obs_cov', 0, 0] = 15099.0
    mod.ssm['transition', 0, 0] = 1
    mod.ssm['selection', 0, 0] = 1
    mod.ssm['state_cov', 0, 0] = 1469.1
    res = mod.filter([])
    actual = res.test_serial_correlation(method='ljungbox', lags=9)[0, 0, -1]
    assert_allclose(actual, [8.84], atol=0.01)
    norm = res.test_normality(method='jarquebera')[0]
    actual = [norm[0], norm[2], norm[3]]
    assert_allclose(actual, [0.05, -0.03, 3.09], atol=0.01)
    actual = res.test_heteroskedasticity(method='breakvar')[0, 0]
    assert_allclose(actual, [0.61], atol=0.01)