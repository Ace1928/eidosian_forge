import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.transform_model import StandardizeTransform
def test_standardize1():
    np.random.seed(123)
    x = 1 + np.random.randn(5, 4)
    transf = StandardizeTransform(x)
    xs1 = transf(x)
    assert_allclose(transf.mean, x.mean(0), rtol=1e-13)
    assert_allclose(transf.scale, x.std(0, ddof=1), rtol=1e-13)
    xs2 = stats.zscore(x, ddof=1)
    assert_allclose(xs1, xs2, rtol=1e-13, atol=1e-20)
    xs4 = transf(2 * x)
    assert_allclose(xs4, (2 * x - transf.mean) / transf.scale, rtol=1e-13, atol=1e-20)
    x2 = 2 * x + np.random.randn(4)
    transf2 = StandardizeTransform(x2)
    xs3 = transf2(x2)
    assert_allclose(xs3, xs1, rtol=1e-13, atol=1e-20)
    x5 = np.column_stack((np.ones(x.shape[0]), x))
    transf5 = StandardizeTransform(x5)
    xs5 = transf5(x5)
    assert_equal(transf5.const_idx, 0)
    assert_equal(xs5[:, 0], np.ones(x.shape[0]))
    assert_allclose(xs5[:, 1:], xs1, rtol=1e-13, atol=1e-20)