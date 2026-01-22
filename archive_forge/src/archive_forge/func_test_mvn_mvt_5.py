import numpy as np
from numpy.testing import assert_almost_equal,  assert_allclose
from statsmodels.sandbox.distributions.multivariate import (
from statsmodels.sandbox.distributions.mv_normal import MVT, MVNormal
def test_mvn_mvt_5(self):
    a, bl = (self.a, self.b)
    df = self.df
    corr2 = self.corr2
    a3 = np.array([0.5, -0.5, 0.5])
    probmvn_R = 0.06910487
    probmvt_R = 0.05797867
    assert_almost_equal(mvstdtprob(a3, a3 + 1, corr2, df), probmvt_R, 4)
    assert_almost_equal(probmvn_R, mvstdnormcdf(a3, a3 + 1, corr2, maxpts=100000, abseps=1e-05), 4)