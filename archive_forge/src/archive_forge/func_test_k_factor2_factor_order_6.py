from statsmodels.compat.pandas import QUARTER_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.statespace import (
@pytest.mark.skip(reason='Monte carlo test, very slow, kept for manual runs')
def test_k_factor2_factor_order_6(reset_randomstate):
    endog_M, endog_Q, factors = gen_k_factor2()
    endog_M_aug = pd.concat([factors, endog_M], axis=1)
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M_aug, endog_quarterly=endog_Q, factor_multiplicities=2, factor_orders=6, idiosyncratic_ar1=False, standardize=False)
    res = mod.fit()
    M = np.kron(np.eye(6), mod['design', :2, :2])
    Mi = np.linalg.inv(M)
    Z = mod['design', :, :12]
    A = mod['transition', :12, :12]
    R = mod['selection', :12, :2]
    Q = mod['state_cov', :2, :2]
    RQR = R @ Q @ R.T
    Z2 = Z @ Mi
    A2 = M @ A @ Mi
    Q2 = M @ RQR @ M.T
    print(Z2.round(2))
    desired = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0], [1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0]])
    assert_allclose(Z2, desired, atol=0.1)
    print(A2.round(2))
    desired = np.array([[0, 0, 0.02, 0, 0.01, -0.03, 0.01, 0.02, 0, -0.01, 0.5, -0.2], [0, 0, 0, 0.02, 0, -0.01, 0, 0, 0, 0.01, 0.1, 0.3], [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0]])
    assert_allclose(A2, desired, atol=0.01)
    print(Q2.round(2))
    desired = np.array([[1.49, 0.21], [0.21, 0.49]])
    assert_allclose(Q2[:2, :2], desired, atol=0.01)
    assert_allclose(Q2[:2, 2:], 0, atol=0.01)
    assert_allclose(Q2[2:, :2], 0, atol=0.01)
    assert_allclose(Q2[2:, 2:], 0, atol=0.01)
    a = res.states.smoothed
    a2 = (M @ a.T.iloc[:12]).T
    assert_allclose(endog_M.values, a2.iloc[:, :2].values, atol=1e-10)