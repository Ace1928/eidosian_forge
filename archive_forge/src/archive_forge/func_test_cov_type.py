import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen
def test_cov_type(self):
    res1 = self.res1
    res2 = self.res2
    assert_equal(self.res1.cov_type, 'HC0')
    assert_equal(self.res1.results_constrained.cov_type, 'HC0')
    cov_kwds = {'description': 'Standard Errors are heteroscedasticity robust (HC0)', 'adjust_df': False, 'use_t': False, 'scaling_factor': None}
    assert_equal(self.res1.cov_kwds, cov_kwds)
    assert_equal(self.res1.cov_kwds, self.res1.results_constrained.cov_kwds)
    params = np.array([0.9681778757470111, 0.43674374940137434, 0.33096260487556745, 0.27415680046693747])
    bse = np.array([0.028126650444581985, 0.03309998456428315, 0.033184585514904545, 0.0342825041305033])
    assert_allclose(res2.params[:self.k_nonzero], params, atol=1e-05)
    assert_allclose(res2.bse[:self.k_nonzero], bse, rtol=5e-06)
    assert_allclose(res1.params[:self.k_nonzero], params, atol=1e-05)
    assert_allclose(res1.bse[:self.k_nonzero], bse, rtol=5e-06)