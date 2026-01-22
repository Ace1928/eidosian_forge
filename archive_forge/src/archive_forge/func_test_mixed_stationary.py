import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.initialization import Initialization
from numpy.testing import assert_allclose, assert_raises
def test_mixed_stationary():
    endog = np.zeros(10)
    mod = sarimax.SARIMAX(endog, order=(2, 1, 0))
    phi = [0.5, -0.2]
    sigma2 = 2.0
    mod.update(np.r_[phi, sigma2])
    init = Initialization(mod.k_states)
    init.set(0, 'diffuse')
    init.set((1, 3), 'stationary')
    desired_cov = np.zeros((3, 3))
    T = np.array([[0.5, 1], [-0.2, 0]])
    Q = np.diag([sigma2, 0])
    desired_cov[1:, 1:] = solve_discrete_lyapunov(T, Q)
    check_initialization(mod, init, [0, 0, 0], np.diag([1, 0, 0]), desired_cov)
    init.clear()
    init.set(0, 'diffuse')
    init.set(1, 'stationary')
    init.set(2, 'approximate_diffuse')
    T = np.array([[0.5]])
    Q = np.diag([sigma2])
    desired_cov = np.diag([0, np.squeeze(solve_discrete_lyapunov(T, Q)), 1000000.0])
    check_initialization(mod, init, [0, 0, 0], np.diag([1, 0, 0]), desired_cov)
    init.clear()
    init.set(0, 'diffuse')
    init.set(1, 'stationary')
    init.set(2, 'stationary')
    desired_cov[2, 2] = 0
    check_initialization(mod, init, [0, 0, 0], np.diag([1, 0, 0]), desired_cov)
    endog = np.zeros((10, 2))
    mod = varmax.VARMAX(endog, order=(1, 0))
    intercept = [1.5, -0.1]
    transition = np.array([[0.5, -0.2], [0.1, 0.8]])
    cov = np.array([[1.2, -0.4], [-0.4, 0.4]])
    tril = np.tril_indices(2)
    params = np.r_[intercept, transition.ravel(), np.linalg.cholesky(cov)[tril]]
    mod.update(params)
    init = Initialization(mod.k_states, 'stationary')
    desired_intercept = np.linalg.solve(np.eye(2) - transition, intercept)
    desired_cov = solve_discrete_lyapunov(transition, cov)
    check_initialization(mod, init, desired_intercept, np.diag([0, 0]), desired_cov)
    init.set(None, 'diffuse')
    check_initialization(mod, init, [0, 0], np.eye(2), np.diag([0, 0]))
    init.unset(None)
    init.set(0, 'stationary')
    init.set(1, 'stationary')
    a, Pinf, Pstar = init(model=mod)
    desired_intercept = [intercept[0] / (1 - transition[0, 0]), intercept[1] / (1 - transition[1, 1])]
    desired_cov = np.diag([cov[0, 0] / (1 - transition[0, 0] ** 2), cov[1, 1] / (1 - transition[1, 1] ** 2)])
    check_initialization(mod, init, desired_intercept, np.diag([0, 0]), desired_cov)