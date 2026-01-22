import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import (KalmanFilter,
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace import tools, initialization
def get_cov_state_revision(t):
    tmp1 = np.zeros((self.k_states, n_updates))
    for i in range(n_updates):
        t_i, k_i = updates_ix[i]
        acov = self.smoothed_state_autocovariance(lag=t - t_i, t=t, extend_kwargs=extend_kwargs)
        Z_i = get_mat('design', t_i)
        tmp1[:, i:i + 1] = acov @ Z_i[k_i:k_i + 1].T
    return tmp1