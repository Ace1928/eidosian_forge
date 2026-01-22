import warnings
import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression
def test_conditional_loglikelihoods():
    endog = np.ones(10)
    mod = markov_autoregression.MarkovAutoregression(endog, k_regimes=2, order=1)
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.ones(9))
    params = np.r_[0.5, 0.5, 2.0, 3.0, 2.0, 0.1, 0.5]
    resid = mod._resid(params)
    conditional_likelihoods = np.exp(-0.5 * resid ** 2 / 2) / np.sqrt(2 * np.pi * 2)
    assert_allclose(mod._conditional_loglikelihoods(params), np.log(conditional_likelihoods))
    endog = np.ones(10)
    mod = markov_autoregression.MarkovAutoregression(endog, k_regimes=3, order=1, switching_variance=True)
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.ones(9))
    params = np.r_[[0.3] * 6, 2.0, 3.0, 4.0, 1.5, 3.0, 4.5, 0.1, 0.5, 0.8]
    mod_conditional_loglikelihoods = mod._conditional_loglikelihoods(params)
    conditional_likelihoods = mod._resid(params)
    conditional_likelihoods[0, :, :] = np.exp(-0.5 * conditional_likelihoods[0, :, :] ** 2 / 1.5) / np.sqrt(2 * np.pi * 1.5)
    assert_allclose(mod_conditional_loglikelihoods[0, :, :], np.log(conditional_likelihoods[0, :, :]))
    conditional_likelihoods[1, :, :] = np.exp(-0.5 * conditional_likelihoods[1, :, :] ** 2 / 3.0) / np.sqrt(2 * np.pi * 3.0)
    assert_allclose(mod_conditional_loglikelihoods[1, :, :], np.log(conditional_likelihoods[1, :, :]))
    conditional_likelihoods[2, :, :] = np.exp(-0.5 * conditional_likelihoods[2, :, :] ** 2 / 4.5) / np.sqrt(2 * np.pi * 4.5)
    assert_allclose(mod_conditional_loglikelihoods[2, :, :], np.log(conditional_likelihoods[2, :, :]))