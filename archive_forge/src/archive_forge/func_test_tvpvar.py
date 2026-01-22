import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels import datasets
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel
def test_tvpvar():
    mod = TVPVAR(endog.iloc[2:])
    sim = mod.simulation_smoother(method='cfa')
    v10 = mod.k_endog + 3
    S10 = np.eye(mod.k_endog)
    vi0 = np.ones(mod.k_states) * 6 / 2
    Si0 = np.ones(mod.k_states) * 0.01 / 2
    initial_obs_cov = np.cov(endog.T)
    initial_state_cov_vars = np.ones(mod.k_states) * 0.01
    mod.update_direct(initial_obs_cov, initial_state_cov_vars)
    res = mod.ssm.smooth()
    variates_1 = results['state_variates'].iloc[:6]
    sim.simulate(variates_1)
    posterior_mean_1 = results['posterior_mean'].iloc[:6]
    assert_allclose(sim.posterior_mean, posterior_mean_1)
    assert_allclose(sim.posterior_mean, res.smoothed_state)
    posterior_cov_1 = np.linalg.inv(results['invP'].iloc[:54])
    assert_allclose(sim.posterior_cov, posterior_cov_1)
    simulated_state_1 = results['beta'].iloc[:6]
    assert_allclose(sim.simulated_state, simulated_state_1)
    fitted = np.matmul(mod['design'].transpose(2, 0, 1), sim.simulated_state.T[..., None])[..., 0]
    resid = mod.endog - fitted
    df = v10 + mod.nobs
    scale = S10 + np.dot(resid.T, resid)
    assert_allclose(df, results['v10'].iloc[:2])
    assert_allclose(scale, results['S10'].iloc[:, :2])
    resid = sim.simulated_state.T[1:] - sim.simulated_state.T[:-1]
    sse = np.sum(resid ** 2, axis=0)
    shapes = vi0 + (mod.nobs - 1) / 2
    scales = Si0 + sse / 2
    assert_allclose(shapes, results['vi0'].values[0, 0])
    assert_allclose(scales, results['Si0'].iloc[:, 0])
    mod.update_direct(results['Omega_11'].iloc[:, :2], results['Omega_22'].iloc[:, 0])
    res = mod.ssm.smooth()
    variates_2 = results['state_variates'].iloc[6:]
    sim.simulate(variates_2)
    posterior_mean_2 = results['posterior_mean'].iloc[6:]
    assert_allclose(sim.posterior_mean, posterior_mean_2)
    assert_allclose(sim.posterior_mean, res.smoothed_state)
    posterior_cov_2 = np.linalg.inv(results['invP'].iloc[54:])
    assert_allclose(sim.posterior_cov, posterior_cov_2)
    simulated_state_2 = results['beta'].iloc[6:]
    assert_allclose(sim.simulated_state, simulated_state_2)
    fitted = np.matmul(mod['design'].transpose(2, 0, 1), sim.simulated_state.T[..., None])[..., 0]
    resid = mod.endog - fitted
    df = v10 + mod.nobs
    scale = S10 + np.dot(resid.T, resid)
    assert_allclose(df, results['v10'].iloc[2:])
    assert_allclose(scale, results['S10'].iloc[:, 2:])
    resid = sim.simulated_state.T[1:] - sim.simulated_state.T[:-1]
    sse = np.sum(resid ** 2, axis=0)
    shapes = vi0 + (mod.nobs - 1) / 2
    scales = Si0 + sse / 2
    assert_allclose(shapes, results['vi0'].values[0, 1])
    assert_allclose(scales, results['Si0'].iloc[:, 1])