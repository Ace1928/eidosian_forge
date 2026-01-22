import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_continuous_cdf(self):
    dens_nm = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.growth], dep_type='c', indep_type='c', bw='normal_reference')
    sm_result = dens_nm.cdf()[0:5]
    R_result = [0.8130492, 0.95046942, 0.86878727, 0.71961748, 0.38685423]
    npt.assert_allclose(sm_result, R_result, atol=0.001)