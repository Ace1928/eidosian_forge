import numpy as np
from statsmodels.multivariate.factor import Factor
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
import warnings
def test_1factor():
    """
    # R code:
    r = 0.4
    p = 4
    ii = seq(0, p-1)
    ii = outer(ii, ii, "-")
    ii = abs(ii)
    cm = r^ii
    fa = factanal(covmat=cm, factors=1)
    print(fa, digits=10)
    """
    r = 0.4
    p = 4
    ii = np.arange(p)
    cm = r ** np.abs(np.subtract.outer(ii, ii))
    fa = Factor(corr=cm, n_factor=1, method='ml')
    rslt = fa.fit()
    if rslt.loadings[0, 0] < 0:
        rslt.loadings[:, 0] *= -1
    uniq = np.r_[0.85290232, 0.60916033, 0.55382266, 0.82610666]
    load = np.asarray([[0.38353316], [0.62517171], [0.66796508], [0.4170052]])
    assert_allclose(load, rslt.loadings, rtol=0.001, atol=0.001)
    assert_allclose(uniq, rslt.uniqueness, rtol=0.001, atol=0.001)
    assert_equal(rslt.df, 2)