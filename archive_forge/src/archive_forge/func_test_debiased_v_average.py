import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_debiased_v_average():
    np.random.seed(435265)
    N = 200
    p = 10
    m = 4
    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)
    db_mod = DistributedModel(m)
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={'alpha': 0.2})
    olsdb = np.linalg.norm(fitOLSdb.params - beta)
    n_mod = DistributedModel(m, estimation_method=_est_regularized_naive, join_method=_join_naive)
    fitOLSn = n_mod.fit(_data_gen(y, X, m), fit_kwds={'alpha': 0.2})
    olsn = np.linalg.norm(fitOLSn.params - beta)
    assert_(olsdb < olsn)
    prob = 1 / (1 + np.exp(-X.dot(beta) + np.random.normal(size=N)))
    y = 1.0 * (prob > 0.5)
    db_mod = DistributedModel(m, model_class=GLM, init_kwds={'family': Binomial()})
    fitGLMdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={'alpha': 0.2})
    glmdb = np.linalg.norm(fitGLMdb.params - beta)
    n_mod = DistributedModel(m, model_class=GLM, init_kwds={'family': Binomial()}, estimation_method=_est_regularized_naive, join_method=_join_naive)
    fitGLMn = n_mod.fit(_data_gen(y, X, m), fit_kwds={'alpha': 0.2})
    glmn = np.linalg.norm(fitGLMn.params - beta)
    assert_(glmdb < glmn)