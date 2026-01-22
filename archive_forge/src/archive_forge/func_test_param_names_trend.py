import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.statespace import varmax, sarimax
from statsmodels.iolib.summary import forg
from .results import results_varmax
def test_param_names_trend():
    endog = np.zeros((3, 2))
    base_names = ['L1.y1.y1', 'L1.y2.y1', 'L1.y1.y2', 'L1.y2.y2', 'sqrt.var.y1', 'sqrt.cov.y1.y2', 'sqrt.var.y2']
    base_params = [0.5, 0, 0, 0.4, 1.0, 0.0, 1.0]
    mod = varmax.VARMAX(endog, order=(1, 0), trend='n')
    desired = base_names
    assert_equal(mod.param_names, desired)
    mod = varmax.VARMAX(endog, order=(1, 0), trend=[1])
    desired = ['intercept.y1', 'intercept.y2'] + base_names
    assert_equal(mod.param_names, desired)
    mod.update([1.2, -0.5] + base_params)
    assert_allclose(mod['state_intercept'], [1.2, -0.5])
    mod = varmax.VARMAX(endog, order=(1, 0), trend=[1, 1])
    desired = ['intercept.y1', 'drift.y1', 'intercept.y2', 'drift.y2'] + base_names
    assert_equal(mod.param_names, desired)
    mod.update([1.2, 0, -0.5, 0] + base_params)
    assert_allclose(mod['state_intercept', 0], 1.2)
    assert_allclose(mod['state_intercept', 1], -0.5)
    mod.update([0, 1, 0, 1.1] + base_params)
    assert_allclose(mod['state_intercept', 0], np.arange(2, 5))
    assert_allclose(mod['state_intercept', 1], 1.1 * np.arange(2, 5))
    mod.update([1.2, 1, -0.5, 1.1] + base_params)
    assert_allclose(mod['state_intercept', 0], 1.2 + np.arange(2, 5))
    assert_allclose(mod['state_intercept', 1], -0.5 + 1.1 * np.arange(2, 5))
    mod = varmax.VARMAX(endog, order=(1, 0), trend=[0, 1])
    desired = ['drift.y1', 'drift.y2'] + base_names
    assert_equal(mod.param_names, desired)
    mod.update([1, 1.1] + base_params)
    assert_allclose(mod['state_intercept', 0], np.arange(2, 5))
    assert_allclose(mod['state_intercept', 1], 1.1 * np.arange(2, 5))
    mod = varmax.VARMAX(endog, order=(1, 0), trend=[1, 0, 1])
    desired = ['intercept.y1', 'trend.2.y1', 'intercept.y2', 'trend.2.y2'] + base_names
    assert_equal(mod.param_names, desired)
    mod.update([1.2, 0, -0.5, 0] + base_params)
    assert_allclose(mod['state_intercept', 0], 1.2)
    assert_allclose(mod['state_intercept', 1], -0.5)
    mod.update([0, 1, 0, 1.1] + base_params)
    assert_allclose(mod['state_intercept', 0], np.arange(2, 5) ** 2)
    assert_allclose(mod['state_intercept', 1], 1.1 * np.arange(2, 5) ** 2)
    mod.update([1.2, 1, -0.5, 1.1] + base_params)
    assert_allclose(mod['state_intercept', 0], 1.2 + np.arange(2, 5) ** 2)
    assert_allclose(mod['state_intercept', 1], -0.5 + 1.1 * np.arange(2, 5) ** 2)