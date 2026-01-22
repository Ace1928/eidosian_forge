import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.initialization import Initialization
from numpy.testing import assert_allclose, assert_raises
def test_global_diffuse():
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    init = Initialization(mod.k_states, 'diffuse')
    check_initialization(mod, init, [0], np.eye(1), np.diag([0]))
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    init = Initialization(mod.k_states, 'diffuse')
    check_initialization(mod, init, [0, 0], np.eye(2), np.diag([0, 0]))