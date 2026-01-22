import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.initialization import Initialization
from numpy.testing import assert_allclose, assert_raises
def check_initialization(mod, init, a_true, Pinf_true, Pstar_true):
    a, Pinf, Pstar = init(model=mod)
    assert_allclose(a, a_true)
    assert_allclose(Pinf, Pinf_true)
    assert_allclose(Pstar, Pstar_true)
    mod.ssm._initialize_representation()
    init._initialize_initialization(prefix=mod.ssm.prefix)
    _statespace = mod.ssm._statespace
    _statespace.initialize(init)
    assert_allclose(np.array(_statespace.initial_state), a_true)
    assert_allclose(np.array(_statespace.initial_diffuse_state_cov), Pinf_true)
    assert_allclose(np.array(_statespace.initial_state_cov), Pstar_true)