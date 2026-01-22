import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
def test_apply_results():
    endog = np.arange(200).reshape(100, 2)
    exog = np.ones(100)
    params = [0.1, -0.2, 1.0, 2.0, 1.0, 1.0, 0.5, 0.1]
    mod1 = dynamic_factor.DynamicFactor(endog[:50], k_factors=1, factor_order=2, exog=exog[:50])
    res1 = mod1.smooth(params)
    mod2 = dynamic_factor.DynamicFactor(endog[50:], k_factors=1, factor_order=2, exog=exog[50:])
    res2 = mod2.smooth(params)
    res3 = res2.apply(endog[:50], exog=exog[:50])
    assert_equal(res1.specification, res3.specification)
    assert_allclose(res3.cov_params_default, res2.cov_params_default)
    for attr in ['nobs', 'llf', 'llf_obs', 'loglikelihood_burn']:
        assert_equal(getattr(res3, attr), getattr(res1, attr))
    for attr in ['filtered_state', 'filtered_state_cov', 'predicted_state', 'predicted_state_cov', 'forecasts', 'forecasts_error', 'forecasts_error_cov', 'standardized_forecasts_error', 'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov', 'scaled_smoothed_estimator', 'scaled_smoothed_estimator_cov', 'smoothing_error', 'smoothed_state', 'smoothed_state_cov', 'smoothed_state_autocov', 'smoothed_measurement_disturbance', 'smoothed_state_disturbance', 'smoothed_measurement_disturbance_cov', 'smoothed_state_disturbance_cov']:
        assert_equal(getattr(res3, attr), getattr(res1, attr))
    assert_allclose(res3.forecast(10, exog=np.ones(10)), res1.forecast(10, exog=np.ones(10)))