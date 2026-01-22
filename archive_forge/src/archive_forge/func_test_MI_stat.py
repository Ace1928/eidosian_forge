import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI
from numpy.testing import assert_allclose, assert_equal
def test_MI_stat():
    np.random.seed(414)
    z = np.random.normal(size=(1000, 3))
    z[:, 0] += 0.5 * z[:, 1]
    exp = [1 / np.sqrt(500), 1 / np.sqrt(1000)]
    fmi = [0.5, 0]
    for j, r in enumerate((0, 0.9999)):
        x = z.copy()
        x[:, 2] = r * x[:, 1] + np.sqrt(1 - r ** 2) * x[:, 2]
        x[0:500, 1] = np.nan

        def model_args(x):
            return (x[:, 0], x[:, 1])
        np.random.seed(2342)
        imp = BayesGaussMI(x.copy())
        mi = MI(imp, sm.OLS, model_args, nrep=100, skip=10)
        r = mi.fit()
        d = np.abs(r.bse[0] - exp[j]) / exp[j]
        assert d < 0.03
        d = np.abs(r.fmi[0] - fmi[j])
        assert d < 0.05