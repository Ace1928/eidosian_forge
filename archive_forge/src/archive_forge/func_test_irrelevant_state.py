from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_equal, assert_allclose
from . import kfas_helpers
@pytest.mark.xfail
def test_irrelevant_state():
    endog = macrodata.infl
    spec = {'freq_seasonal': [{'period': 8, 'harmonics': 6}, {'period': 36, 'harmonics': 6}]}
    mod = UnobservedComponents(endog, 'llevel', **spec)
    mod.ssm.initialization = Initialization(mod.k_states, 'approximate_diffuse')
    res = mod.smooth([3.4, 7.2, 0.01, 0.01])
    mod2 = UnobservedComponents(endog, 'llevel', **spec)
    mod2.ssm.filter_univariate = True
    mod2.ssm.initialization = Initialization(mod2.k_states, 'diffuse')
    res2 = mod2.smooth([3.4, 7.2, 0.01, 0.01])
    assert_allclose(res.filtered_state[0, 25:], res2.filtered_state[0, 25:], atol=1e-05)