import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_pdf_mixeddata_CV_ML(self):
    dens_ml = nparam.KDEMultivariate(data=[self.c1, self.o, self.c2], var_type='coc', bw='cv_ml')
    R_bw = [1.021563, 2.806409e-14, 0.5142077]
    npt.assert_allclose(dens_ml.bw, R_bw, atol=0.1, rtol=0.1)