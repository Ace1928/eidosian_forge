from statsmodels.compat.pandas import QUARTER_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.statespace import (
def simulate_k_factors3_blocks2(nobs=1000, idiosyncratic_ar1=False):
    ix = pd.period_range(start='2000-01', periods=1, freq='M')
    endog = pd.DataFrame(np.zeros((1, 2)), columns=['f1', 'f2'], index=ix)
    mod_f_12 = varmax.VARMAX(endog, order=(1, 0), trend='n')
    params = [0.5, 0.1, -0.2, 0.9, 1.0, 0, 1.0]
    f_12 = mod_f_12.simulate(params, nobs)
    endog = pd.Series([0], name='f3', index=ix)
    mod_f_3 = sarimax.SARIMAX(endog, order=(2, 0, 0))
    params = [0.7, 0.1, 1.0]
    f_3 = mod_f_3.simulate(params, nobs)
    f = pd.concat([f_12, f_3], axis=1)
    k_endog = 8
    design = np.zeros((k_endog, 3))
    design[0] = [1.0, 1.0, 1.0]
    design[1] = [0.5, -0.8, 0.0]
    design[2] = [1.0, 0.0, 0.0]
    design[3] = [0.2, 0.0, -0.1]
    design[4] = [0.5, 0.0, 0.0]
    design[5] = [-0.2, 0.0, 0.0]
    design[6] = [1.0, 1.0, 1.0]
    design[7] = [-1.0, 0.0, 0.0]
    rho = np.array([0.5, 0.2, -0.1, 0.0, 0.4, 0.9, 0.05, 0.05])
    if not idiosyncratic_ar1:
        rho *= 0.0
    eps = [lfilter([1], [1, -rho[i]], np.random.normal(size=nobs)) for i in range(k_endog)]
    endog = (design @ f.T).T + eps
    endog.columns = [f'y{i + 1}' for i in range(k_endog)]
    tmp1 = design.ravel()
    tmp2 = np.linalg.cholesky(mod_f_12['state_cov'])
    tmp3 = rho if idiosyncratic_ar1 else []
    true = np.r_[tmp1[tmp1 != 0], mod_f_12['transition', :2, :].ravel(), mod_f_3['transition', :, 0], tmp2[np.tril_indices_from(tmp2)], mod_f_3['state_cov', 0, 0], tmp3, [1] * k_endog]
    ix = pd.period_range(endog.index[0] - 1, endog.index[-1], freq='M')
    levels_M = 1 + endog.reindex(ix) / 100
    levels_M.iloc[0] = 100
    levels_M = levels_M.cumprod()
    log_levels_M = np.log(levels_M) * 100
    log_levels_Q = np.log(levels_M)
    log_levels_Q.index = log_levels_Q.index.to_timestamp()
    log_levels_Q = log_levels_Q.resample(QUARTER_END).sum().iloc[:-1] * 100
    log_levels_Q.index = log_levels_Q.index.to_period()
    endog_M = log_levels_M.iloc[:, :7].diff().iloc[1:]
    endog_Q = log_levels_Q.iloc[:, 7:].diff().iloc[2:]
    factor_names = np.array(['global', 'second', 'third'])
    factors = {endog.columns[i]: factor_names[design[i] != 0] for i in range(k_endog)}
    factor_orders = {('global', 'second'): 1, 'third': 2}
    return (endog_M, endog_Q, log_levels_M, log_levels_Q, factors, factor_orders, true, f)