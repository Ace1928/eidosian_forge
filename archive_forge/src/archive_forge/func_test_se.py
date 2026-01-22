import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import add_constant
def test_se(self):
    res1 = self.res1
    res2 = self.res2
    assert_equal(res1.df_resid, res2.df_resid)
    assert_allclose(res1.scale, res2.scale, rtol=1e-10)
    assert_allclose(res1.bse[self.idx_p_uc], res2.bse, rtol=1e-10)
    assert_allclose(res1.cov_params()[self.idx_p_uc[:, None], self.idx_p_uc], res2.cov_params(), rtol=5e-09, atol=1e-15)