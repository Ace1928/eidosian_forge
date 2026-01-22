import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_repeat_partition():
    np.random.seed(435265)
    N = 200
    p = 10
    m = 1
    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)

    def _rep_data_gen(endog, exog, partitions):
        """partitions data"""
        n_exog = exog.shape[0]
        n_part = np.ceil(n_exog / partitions)
        ii = 0
        while ii < n_exog:
            yield (endog, exog)
            ii += int(n_part)
    nv_mod = DistributedModel(m, estimation_method=_est_regularized_naive, join_method=_join_naive)
    fitOLSnv = nv_mod.fit(_rep_data_gen(y, X, m), fit_kwds={'alpha': 0.1})
    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit_regularized(alpha=0.1)
    assert_allclose(fitOLSnv.params, fitOLS.params)