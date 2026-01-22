import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
@pytest.mark.slow
@pytest.mark.xfail(reason='Test does not make much sense - always passes with very small bw.')
def test_mfx_nonlinear_ll_cvls(self, file_name='RegData.csv'):
    nobs = 200
    np.random.seed(1234)
    C1 = np.random.normal(size=(nobs,))
    C2 = np.random.normal(2, 1, size=(nobs,))
    C3 = np.random.beta(0.5, 0.2, size=(nobs,))
    noise = np.random.normal(size=(nobs,))
    b0 = 3
    b1 = 1.2
    b3 = 2.3
    Y = b0 + b1 * C1 * C2 + b3 * C3 + noise
    model = nparam.KernelReg(endog=[Y], exog=[C1, C2, C3], reg_type='ll', var_type='ccc', bw='cv_ls')
    sm_bw = model.bw
    sm_mean, sm_mfx = model.fit()
    sm_R2 = model.r_squared()
    mfx1 = b1 * C2
    mfx2 = b1 * C1
    npt.assert_allclose(sm_mean, Y, rtol=0.2)
    npt.assert_allclose(sm_mfx[:, 0], mfx1, rtol=0.2)
    npt.assert_allclose(sm_mfx[0:10, 1], mfx2[0:10], rtol=0.2)