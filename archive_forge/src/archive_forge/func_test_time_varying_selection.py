from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
@pytest.mark.smoke
def test_time_varying_selection(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))
    mod['obs_cov'] = mod['obs_cov', :, :, 0]
    mod['state_cov'] = mod['state_cov', :, :, 0]
    assert_equal(mod['obs_cov'].shape, (mod.k_endog, mod.k_endog))
    assert_equal(mod['state_cov'].shape, (mod.ssm.k_posdef, mod.ssm.k_posdef))
    mod.simulate([], 10)