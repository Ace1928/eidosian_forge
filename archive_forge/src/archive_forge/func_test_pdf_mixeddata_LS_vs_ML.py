import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
@pytest.mark.slow
def test_pdf_mixeddata_LS_vs_ML(self):
    dens_ls = nparam.KDEMultivariate(data=[self.c1, self.o, self.o2], var_type='coo', bw='cv_ls')
    dens_ml = nparam.KDEMultivariate(data=[self.c1, self.o, self.o2], var_type='coo', bw='cv_ml')
    npt.assert_allclose(dens_ls.bw, dens_ml.bw, atol=0, rtol=0.5)