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
def model_local_linear_trend(endog=None, params=None, direct=False):
    if endog is None:
        y1 = 10.2394
        y2 = 4.2039
        y3 = 6.123123
        endog = np.r_[y1, y2, y3, [1] * 7]
    if params is None:
        params = [1.993, 8.253, 2.334]
    sigma2_y, sigma2_mu, sigma2_beta = params
    if direct:
        mod = None
        ssm = KalmanSmoother(k_endog=1, k_states=2, k_posdef=2)
        ssm.bind(endog)
        init = Initialization(ssm.k_states, initialization_type='diffuse')
        ssm.initialize(init)
        ssm['design', 0, 0] = 1
        ssm['obs_cov', 0, 0] = sigma2_y
        ssm['transition'] = np.array([[1, 1], [0, 1]])
        ssm['selection'] = np.eye(2)
        ssm['state_cov'] = np.diag([sigma2_mu, sigma2_beta])
    else:
        mod = UnobservedComponents(endog, 'lltrend')
        mod.update(params)
        ssm = mod.ssm
        ssm.initialize(Initialization(ssm.k_states, 'diffuse'))
    return (mod, ssm)