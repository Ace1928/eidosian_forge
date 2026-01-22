import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_pdf_ordered(self):
    dens = nparam.KDEMultivariate(data=[self.oecd], var_type='o', bw='cv_ls')
    sm_result = np.squeeze(dens.pdf()[0:5])
    R_result = [0.7236395, 0.7236395, 0.2763605, 0.2763605, 0.7236395]
    npt.assert_allclose(sm_result, R_result, atol=0.1)