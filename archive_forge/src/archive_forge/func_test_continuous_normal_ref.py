import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_continuous_normal_ref(self):
    dens_nm = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.growth], dep_type='c', indep_type='c', bw='normal_reference')
    sm_result = dens_nm.bw
    R_result = [1.283532, 0.01535401]
    npt.assert_allclose(sm_result, R_result, atol=0.1)
    dens_nm2 = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.growth], dep_type='c', indep_type='c', bw=None)
    assert_allclose(dens_nm2.bw, dens_nm.bw, rtol=1e-10)
    assert_equal(dens_nm2._bw_method, 'normal_reference')
    repr(dens_nm2)