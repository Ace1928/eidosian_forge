from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_known_initialization():
    np.random.seed(38947)
    nobs = 100
    eps = np.random.normal(size=nobs)
    eps1 = np.zeros(nobs)
    eps2 = np.zeros(nobs)
    eps2[49] = 1
    eps3 = np.zeros(nobs)
    eps3[50:] = 1
    mod = sarimax.SARIMAX([0], order=(1, 0, 0))
    mod.ssm.initialize_known([100], [[0]])
    actual = mod.simulate([0.5, 1.0], nobs, state_shocks=eps1)
    assert_allclose(actual, 100 * 0.5 ** np.arange(nobs))
    mod = structural.UnobservedComponents([0], 'local level')
    mod.ssm.initialize_known([100], [[0]])
    actual = mod.simulate([1.0, 1.0], nobs, measurement_shocks=eps, state_shocks=eps2)
    assert_allclose(actual, 100 + eps + eps3)
    transition = np.diag([0.5, 0.2])
    mod = varmax.VARMAX([[0, 0]], order=(1, 0), trend='n')
    mod.initialize_known([100, 50], np.diag([0, 0]))
    actual = mod.simulate(np.r_[transition.ravel(), 1.0, 0, 1.0], nobs, measurement_shocks=np.c_[eps1, eps1], state_shocks=np.c_[eps1, eps1])
    assert_allclose(actual, np.c_[100 * 0.5 ** np.arange(nobs), 50 * 0.2 ** np.arange(nobs)])
    mod = dynamic_factor.DynamicFactor([[0, 0]], k_factors=1, factor_order=1)
    mod.initialize_known([100], [[0]])
    actual = mod.simulate([0.8, 0.2, 1.0, 1.0, 0.5], nobs, measurement_shocks=np.c_[eps1, eps1], state_shocks=eps1)
    tmp = 100 * 0.5 ** np.arange(nobs)
    assert_allclose(actual, np.c_[0.8 * tmp, 0.2 * tmp])