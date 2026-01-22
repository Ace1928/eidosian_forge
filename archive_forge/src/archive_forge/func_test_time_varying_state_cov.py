from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_time_varying_state_cov(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))
    mod['obs_cov'] = mod['obs_cov', :, :, 0] * 0
    mod['selection'] = mod['selection', :, :, 0]
    mod['state_intercept', :] = 0
    mod['state_cov'] = np.zeros((mod.ssm.k_posdef, mod.ssm.k_posdef, mod.nobs))
    mod['state_cov', ..., -1] = np.eye(mod.ssm.k_posdef)
    assert_equal(mod['obs_cov'].shape, (mod.k_endog, mod.k_endog))
    assert_equal(mod['selection'].shape, (mod.k_states, mod.ssm.k_posdef))
    sim = mod.simulate([], 10)
    assert_allclose(sim, mod['obs_intercept'].T)