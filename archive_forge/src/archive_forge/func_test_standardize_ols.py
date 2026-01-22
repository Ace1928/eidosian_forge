import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.transform_model import StandardizeTransform
def test_standardize_ols():
    np.random.seed(123)
    nobs = 20
    x = 1 + np.random.randn(nobs, 4)
    exog = np.column_stack((np.ones(nobs), x))
    endog = exog.sum(1) + np.random.randn(nobs)
    res2 = OLS(endog, exog).fit()
    transf = StandardizeTransform(exog)
    exog_st = transf(exog)
    res1 = OLS(endog, exog_st).fit()
    params = transf.transform_params(res1.params)
    assert_allclose(params, res2.params, rtol=1e-13)