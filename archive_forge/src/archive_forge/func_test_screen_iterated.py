import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.base._screening import VariableScreening
def test_screen_iterated():
    np.random.seed(987865)
    nobs, k_nonzero = (100, 5)
    x = (np.random.rand(nobs, k_nonzero - 1) + 1.0 * (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
    x *= 1.2
    x = (x - x.mean(0)) / x.std(0)
    x = np.column_stack((np.ones(nobs), x))
    beta = 1.0 / np.arange(1, k_nonzero + 1)
    beta = np.sqrt(beta)
    linpred = x.dot(beta)
    mu = np.exp(linpred)
    y = np.random.poisson(mu)
    common = x[:, 1:].sum(1)[:, None]
    x_nonzero = x

    def exog_iterator():
        k_vars = 100
        n_batches = 6
        for i in range(n_batches):
            x = (0.05 * common + np.random.rand(nobs, k_vars) + 1.0 * (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
            x *= 1.2
            if i < k_nonzero - 1:
                x[:, 10] = x_nonzero[:, i + 1]
            x = (x - x.mean(0)) / x.std(0)
            yield x
    dummy = np.ones(nobs)
    dummy[:nobs // 2] = 0
    exog_keep = np.column_stack((np.ones(nobs), dummy))
    for k in [1, 2]:
        mod_initial = PoissonPenalized(y, exog_keep[:, :k], pen_weight=nobs * 500)
        screener = VariableScreening(mod_initial)
        screener.k_max_add = 30
        final = screener.screen_exog_iterator(exog_iterator())
        names = ['var0_10', 'var1_10', 'var2_10', 'var3_10']
        assert_equal(final.exog_final_names, names)
        idx_full = np.array([[0, 10], [1, 10], [2, 10], [3, 10]], dtype=np.int64)
        assert_equal(final.idx_nonzero_batches, idx_full)