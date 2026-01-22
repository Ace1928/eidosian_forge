import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_extend_by_one():
    endog = np.arange(100)
    exog = np.ones_like(endog)
    params = [1.0, 1.0, 0.1, 1.0]
    mod1 = sarimax.SARIMAX(endog, exog=exog, order=(1, 0, 0), trend='t')
    res1 = mod1.smooth(params)
    mod2 = sarimax.SARIMAX(endog[:-1], exog=exog[:-1], order=(1, 0, 0), trend='t')
    res2 = mod2.smooth(params)
    res3 = res2.extend(endog[-1:], exog=exog[-1:])
    assert_allclose(res3.llf_obs, res1.llf_obs[-1:])
    for attr in ['filtered_state', 'filtered_state_cov', 'predicted_state', 'predicted_state_cov', 'forecasts', 'forecasts_error', 'forecasts_error_cov', 'standardized_forecasts_error', 'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov', 'scaled_smoothed_estimator', 'scaled_smoothed_estimator_cov', 'smoothing_error', 'smoothed_state', 'smoothed_state_cov', 'smoothed_state_autocov', 'smoothed_measurement_disturbance', 'smoothed_state_disturbance', 'smoothed_measurement_disturbance_cov', 'smoothed_state_disturbance_cov']:
        desired = getattr(res1, attr)
        if desired is not None:
            desired = desired[..., 99:]
        assert_equal(getattr(res3, attr), desired)
    assert_allclose(res3.forecast(10, exog=np.ones(10) * 2), res1.forecast(10, exog=np.ones(10) * 2))
    with pytest.raises(ValueError, match='Cloning a model with an exogenous'):
        res2.extend(endog[-1:])