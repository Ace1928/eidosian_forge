import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_calc_grad():
    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    beta = np.random.normal(size=3)
    mod = OLS(y, X)
    grad = _calc_grad(mod, beta, 0.01, 1, {})
    assert_allclose(grad, np.array([19.75816, -6.62307, 7.324644]), atol=1e-06, rtol=0)