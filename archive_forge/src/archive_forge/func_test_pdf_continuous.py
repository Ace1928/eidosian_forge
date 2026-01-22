import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_pdf_continuous(self):
    bw_cv_ml = np.array([0.010043, 12095254.7])
    dens = nparam.KDEMultivariateConditional(endog=[self.growth], exog=[self.Italy_gdp], dep_type='c', indep_type='c', bw=bw_cv_ml)
    sm_result = np.squeeze(dens.pdf()[0:5])
    R_result = [11.97964, 12.7329, 13.23037, 13.46438, 12.22779]
    npt.assert_allclose(sm_result, R_result, atol=0.001)