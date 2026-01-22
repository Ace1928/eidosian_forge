import numpy as np
from statsmodels.discrete.conditional_models import (
from statsmodels.tools.numdiff import approx_fprime
from numpy.testing import assert_allclose
import pandas as pd
def test_lasso_logistic():
    np.random.seed(3423948)
    n = 200
    groups = np.arange(10)
    groups = np.kron(groups, np.ones(n // 10))
    group_effects = np.random.normal(size=10)
    group_effects = np.kron(group_effects, np.ones(n // 10))
    x = np.random.normal(size=(n, 4))
    params = np.r_[0, 0, 1, 0]
    lin_pred = np.dot(x, params) + group_effects
    mean = 1 / (1 + np.exp(-lin_pred))
    y = (np.random.uniform(size=n) < mean).astype(int)
    model0 = ConditionalLogit(y, x, groups=groups)
    result0 = model0.fit()
    model1 = ConditionalLogit(y, x, groups=groups)
    result1 = model1.fit_regularized(L1_wt=0, alpha=0)
    assert_allclose(result0.params, result1.params, rtol=0.001)
    model2 = ConditionalLogit(y, x, groups=groups)
    result2 = model2.fit_regularized(L1_wt=1, alpha=0.05)
    assert_allclose(result2.params, np.r_[0, 0, 0.55235152, 0], rtol=0.0001)
    df = pd.DataFrame({'y': y, 'x1': x[:, 0], 'x2': x[:, 1], 'x3': x[:, 2], 'x4': x[:, 3], 'groups': groups})
    fml = 'y ~ 0 + x1 + x2 + x3 + x4'
    model3 = ConditionalLogit.from_formula(fml, groups='groups', data=df)
    result3 = model3.fit_regularized(L1_wt=1, alpha=0.05)
    assert_allclose(result2.params, result3.params)