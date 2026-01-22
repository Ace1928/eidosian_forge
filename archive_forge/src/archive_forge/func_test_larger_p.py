import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_larger_p():
    np.random.seed(435265)
    N = 40
    p = 40
    m = 5
    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)
    db_mod = DistributedModel(m)
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={'alpha': 0.1})
    assert_equal(np.sum(np.isnan(fitOLSdb.params)), 0)
    nv_mod = DistributedModel(m, estimation_method=_est_regularized_naive, join_method=_join_naive)
    fitOLSnv = nv_mod.fit(_data_gen(y, X, m), fit_kwds={'alpha': 0.1})
    assert_equal(np.sum(np.isnan(fitOLSnv.params)), 0)