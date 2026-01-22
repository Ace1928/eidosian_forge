import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_non_zero_params():
    np.random.seed(435265)
    N = 200
    p = 10
    m = 5
    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)
    db_mod = DistributedModel(m, join_kwds={'threshold': 0.13})
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={'alpha': 0.1})
    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit_regularized(alpha=0.1)
    nz_params_db = 1 * (fitOLSdb.params != 0)
    nz_params_ols = 1 * (fitOLS.params != 0)
    assert_allclose(nz_params_db, nz_params_ols)