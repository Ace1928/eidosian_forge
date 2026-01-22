import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
@pytest.mark.slow
def test_continuous_cvml_efficient(self):
    nobs = 500
    np.random.seed(12345)
    ovals = np.random.binomial(2, 0.5, size=(nobs,))
    C1 = np.random.normal(size=(nobs,))
    noise = np.random.normal(size=(nobs,))
    b0 = 3
    b1 = 1.2
    b2 = 3.7
    Y = b0 + b1 * C1 + b2 * ovals + noise
    dens_efficient = nparam.KDEMultivariateConditional(endog=[Y], exog=[C1], dep_type='c', indep_type='c', bw='cv_ml', defaults=nparam.EstimatorSettings(efficient=True, n_sub=50))
    bw_expected = np.array([0.73387, 0.43715])
    npt.assert_allclose(dens_efficient.bw, bw_expected, atol=0, rtol=0.001)