import numpy as np
import pandas as pd
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from numpy.testing import assert_array_almost_equal, assert_raises
import patsy
def test_from_formula_vs_no_formula():
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data)
    r = mod.fit(method='svd')
    r0 = r.mv_test()
    endog, exog = patsy.dmatrices('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data, return_type='dataframe')
    L = np.array([[1, 0, 0, 0, 0, 0]])
    r = _MultivariateOLS(endog, exog).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Intercept', L, None]])
    assert_array_almost_equal(r1['Intercept']['stat'].values, r0['Intercept']['stat'].values, decimal=6)
    r = _MultivariateOLS(endog.values, exog.values).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Intercept', L, None]])
    assert_array_almost_equal(r1['Intercept']['stat'].values, r0['Intercept']['stat'].values, decimal=6)
    L = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    r1 = r.mv_test(hypotheses=[['Drug', L, None]])
    r = _MultivariateOLS(endog, exog).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Drug', L, None]])
    assert_array_almost_equal(r1['Drug']['stat'].values, r0['Drug']['stat'].values, decimal=6)
    r = _MultivariateOLS(endog.values, exog.values).fit(method='svd')
    r1 = r.mv_test(hypotheses=[['Drug', L, None]])
    assert_array_almost_equal(r1['Drug']['stat'].values, r0['Drug']['stat'].values, decimal=6)