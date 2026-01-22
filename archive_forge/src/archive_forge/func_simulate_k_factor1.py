from statsmodels.compat.pandas import QUARTER_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.statespace import (
def simulate_k_factor1(nobs=1000):
    mod_sim = dynamic_factor.DynamicFactor(np.zeros((1, 4)), k_factors=1, factor_order=1, error_order=1)
    loadings = [1.0, -0.75, 0.25, -0.3, 0.5]
    p = np.r_[loadings[:mod_sim.k_endog], [10] * mod_sim.k_endog, 0.5, [0.0] * mod_sim.k_endog]
    ix = pd.period_range(start='1935-01', periods=nobs, freq='M')
    endog = pd.DataFrame(mod_sim.simulate(p, nobs), index=ix)
    true = pd.Series(p, index=mod_sim.param_names)
    ix = pd.period_range(start=endog.index[0] - 1, end=endog.index[-1], freq=endog.index.freq)
    levels_M = 1 + endog.reindex(ix) / 100
    levels_M.iloc[0] = 100
    levels_M = levels_M.cumprod()
    log_levels_M = np.log(levels_M) * 100
    log_levels_Q = np.log(levels_M)
    log_levels_Q.index = log_levels_Q.index.to_timestamp()
    log_levels_Q = log_levels_Q.resample(QUARTER_END).sum().iloc[:-1] * 100
    log_levels_Q.index = log_levels_Q.index.to_period()
    endog_M = log_levels_M.iloc[:, :3].diff()
    endog_Q = log_levels_Q.iloc[:, 3:].diff()
    return (endog_M, endog_Q, log_levels_M, log_levels_Q, true)