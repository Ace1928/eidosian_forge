import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.statespace import structural
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.results import results_structural
def test_matrices_somewhat_complicated_model():
    values = dta.copy()
    model = UnobservedComponents(values['unemp'], level='lltrend', freq_seasonal=[{'period': 4}, {'period': 9, 'harmonics': 3}], cycle=True, cycle_period_bounds=[2, 30], damped_cycle=True, stochastic_freq_seasonal=[True, False], stochastic_cycle=True)
    params = [1, 3, 4, 5, 6, 2 * np.pi / 30.0, 0.9]
    model.update(params)
    assert_equal(model.k_states, 2 + 4 + 6 + 2)
    assert_equal(model.k_state_cov, 2 + 1 + 0 + 1)
    assert_equal(model.loglikelihood_burn, 2 + 4 + 6 + 2)
    assert_allclose(model.ssm.k_posdef, 2 + 4 + 0 + 2)
    assert_equal(model.k_params, len(params))
    expected_design = np.r_[[1, 0], [1, 0, 1, 0], [1, 0, 1, 0, 1, 0], [1, 0]].reshape(1, 14)
    assert_allclose(model.ssm.design[:, :, 0], expected_design)
    expected_transition = __direct_sum([np.array([[1, 1], [0, 1]]), np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]), np.array([[np.cos(2 * np.pi * 1 / 9.0), np.sin(2 * np.pi * 1 / 9.0), 0, 0, 0, 0], [-np.sin(2 * np.pi * 1 / 9.0), np.cos(2 * np.pi * 1 / 9.0), 0, 0, 0, 0], [0, 0, np.cos(2 * np.pi * 2 / 9.0), np.sin(2 * np.pi * 2 / 9.0), 0, 0], [0, 0, -np.sin(2 * np.pi * 2 / 9.0), np.cos(2 * np.pi * 2 / 9.0), 0, 0], [0, 0, 0, 0, np.cos(2 * np.pi / 3.0), np.sin(2 * np.pi / 3.0)], [0, 0, 0, 0, -np.sin(2 * np.pi / 3.0), np.cos(2 * np.pi / 3.0)]]), np.array([[0.9 * np.cos(2 * np.pi / 30.0), 0.9 * np.sin(2 * np.pi / 30.0)], [-0.9 * np.sin(2 * np.pi / 30.0), 0.9 * np.cos(2 * np.pi / 30.0)]])])
    assert_allclose(model.ssm.transition[:, :, 0], expected_transition, atol=1e-07)
    expected_selection = np.zeros((14, 14 - 6))
    expected_selection[0:2, 0:2] = np.eye(2)
    expected_selection[2:6, 2:6] = np.eye(4)
    expected_selection[-2:, -2:] = np.eye(2)
    assert_allclose(model.ssm.selection[:, :, 0], expected_selection)
    expected_state_cov = __direct_sum([np.diag(params[1:3]), np.eye(4) * params[3], np.eye(2) * params[4]])
    assert_allclose(model.ssm.state_cov[:, :, 0], expected_state_cov)