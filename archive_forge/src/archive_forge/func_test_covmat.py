import numpy as np
from numpy.testing import assert_allclose, assert_equal  #noqa
from statsmodels.stats import weightstats
import statsmodels.stats.multivariate as smmv  # pytest cannot import test_xxx
from statsmodels.stats.multivariate import confint_mvmean_fromstats
from statsmodels.tools.testing import Holder
def test_covmat(self):
    cov, nobs = (self.cov, self.nobs)
    p_chi2 = 0.4837049015162541
    chi2 = 5.481422374989864
    cov_null = np.array([[30, 15, 0], [15, 20, 0], [0, 0, 10]])
    stat, pv = smmv.test_cov(cov, nobs, cov_null)
    assert_allclose(stat, chi2, rtol=1e-07)
    assert_allclose(pv, p_chi2, rtol=1e-06)