import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.initialization import Initialization
from numpy.testing import assert_allclose, assert_raises
def test_global_approximate_diffuse():
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    init = Initialization(mod.k_states, 'approximate_diffuse')
    check_initialization(mod, init, [0], np.diag([0]), np.eye(1) * 1000000.0)
    init = Initialization(mod.k_states, 'approximate_diffuse', constant=[1.2])
    check_initialization(mod, init, [1.2], np.diag([0]), np.eye(1) * 1000000.0)
    init = Initialization(mod.k_states, 'approximate_diffuse', approximate_diffuse_variance=10000000000.0)
    check_initialization(mod, init, [0], np.diag([0]), np.eye(1) * 10000000000.0)
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    init = Initialization(mod.k_states, 'approximate_diffuse')
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), np.eye(2) * 1000000.0)
    init = Initialization(mod.k_states, 'approximate_diffuse', constant=[1.2, -0.2])
    check_initialization(mod, init, [1.2, -0.2], np.diag([0, 0]), np.eye(2) * 1000000.0)
    init = Initialization(mod.k_states, 'approximate_diffuse', approximate_diffuse_variance=10000000000.0)
    check_initialization(mod, init, [0, 0], np.diag([0, 0]), np.eye(2) * 10000000000.0)