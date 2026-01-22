import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI
from numpy.testing import assert_allclose, assert_equal
def test_MI():
    np.random.seed(414)
    x = np.random.normal(size=(200, 4))
    x[[1, 3, 9], 0] = np.nan
    x[[1, 4, 3], 1] = np.nan
    x[[2, 11, 21], 2] = np.nan
    x[[11, 22, 99], 3] = np.nan

    def model_args_fn(x):
        if type(x) is np.ndarray:
            return (x[:, 0], x[:, 1:])
        else:
            return (x.iloc[:, 0].values, x.iloc[:, 1:].values)
    for j in (0, 1):
        np.random.seed(2342)
        imp = BayesGaussMI(x.copy())
        mi = MI(imp, sm.OLS, model_args_fn, burn=0)
        r = mi.fit()
        r.summary()
        assert_allclose(r.params, np.r_[-0.05347919, -0.02479701, 0.10075517], 0.25, 0)
        c = np.asarray([[0.00418232, 0.00029746, -0.00035057], [0.00029746, 0.00407264, 0.00019496], [-0.00035057, 0.00019496, 0.00509413]])
        assert_allclose(r.cov_params(), c, 0.3, 0)
        x = pd.DataFrame(x)