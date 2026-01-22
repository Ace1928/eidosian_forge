import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_join_naive():
    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    mod = OLS(y, X)
    res_l = []
    for i in range(2):
        res = _est_regularized_naive(mod, i, 2, fit_kwds={'alpha': 0.1})
        res_l.append(res)
    joined = _join_naive(res_l)
    assert_allclose(joined, np.array([-0.020757, 0.0, 0.0]), atol=1e-06, rtol=0)
    mod = GLM(y, X, family=Binomial())
    res_l = []
    for i in range(2):
        res = _est_regularized_naive(mod, i, 2, fit_kwds={'alpha': 0.1})
        res_l.append(res)
    joined = _join_naive(res_l)
    assert_allclose(joined, np.array([0.0, 0.0, 0.0]), atol=1e-06, rtol=0)