import numpy as np
import pytest
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_allclose
@pytest.mark.parametrize('missing', [None, 'init', 'mixed', 'all'])
@pytest.mark.parametrize('periods', [np.s_[0], np.s_[4:6], np.s_[:]])
def test_dynamic_factor(missing, periods):
    endog = np.array([[0.5, 1.2, -0.2, 0.3, -0.1, 0.4, 1.4, 0.9], [-0.2, -0.3, -0.1, 0.1, 0.01, 0.05, -0.13, -0.2]]).T
    exog = np.ones_like(endog[:, 0])
    if missing == 'init':
        endog[0:2, :] = np.nan
    elif missing == 'mixed':
        endog[2:4, 0] = np.nan
        endog[3:6, 1] = np.nan
    elif missing == 'all':
        endog[:] = np.nan
    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=2, exog=exog)
    mod.update([1.0, -0.5, 0.3, -0.1, 1.2, 2.3, 0.5, 0.1])
    check_filter_output(mod, periods)
    check_smoother_output(mod, periods)