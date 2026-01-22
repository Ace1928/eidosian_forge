import os
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching import (markov_switching,
def test_exog_tvtp():
    exog = np.ones_like(fedfunds)
    mod1 = markov_regression.MarkovRegression(fedfunds, k_regimes=2)
    mod2 = markov_regression.MarkovRegression(fedfunds, k_regimes=2, exog_tvtp=exog)
    params = np.r_[0.98209618, 0.05036498, 3.70877542, 9.55676298, 4.44181911]
    params_tvtp = params.copy()
    params_tvtp[0] = np.squeeze(mod2._untransform_logistic(np.r_[0.0], np.r_[1 - params[0]]))
    params_tvtp[1] = np.squeeze(mod2._untransform_logistic(np.r_[0.0], np.r_[1 - params[1]]))
    res1 = mod1.smooth(params)
    res2 = mod2.smooth(params_tvtp)
    assert_allclose(res2.llf_obs, res1.llf_obs)
    assert_allclose(res2.regime_transition - res1.regime_transition, 0, atol=1e-15)
    assert_allclose(res2.predicted_joint_probabilities, res1.predicted_joint_probabilities)
    assert_allclose(res2.filtered_joint_probabilities, res1.filtered_joint_probabilities)
    assert_allclose(res2.smoothed_joint_probabilities, res1.smoothed_joint_probabilities)