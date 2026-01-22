import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import block_diag
import pytest
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.gam.smooth_basis import (
from statsmodels.gam.generalized_additive_model import (
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_penalties import (UnivariateGamPenalty,
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Gaussian
from statsmodels.genmod.generalized_linear_model import lm
def test_multivariate_penalty():
    alphas = [1, 2]
    weights = [1, 1]
    np.random.seed(1)
    x, y, pol = multivariate_sample_data()
    univ_pol1 = UnivariatePolynomialSmoother(x[:, 0], degree=pol.degrees[0])
    univ_pol2 = UnivariatePolynomialSmoother(x[:, 1], degree=pol.degrees[1])
    gp1 = UnivariateGamPenalty(alpha=alphas[0], univariate_smoother=univ_pol1)
    gp2 = UnivariateGamPenalty(alpha=alphas[1], univariate_smoother=univ_pol2)
    with pytest.warns(UserWarning, match='weights is currently ignored'):
        mgp = MultivariateGamPenalty(multivariate_smoother=pol, alpha=alphas, weights=weights)
    for i in range(10):
        params1 = np.random.randint(-3, 3, pol.smoothers[0].dim_basis)
        params2 = np.random.randint(-3, 3, pol.smoothers[1].dim_basis)
        params = np.concatenate([params1, params2])
        c1 = gp1.func(params1)
        c2 = gp2.func(params2)
        c = mgp.func(params)
        assert_allclose(c, c1 + c2, atol=1e-10, rtol=1e-10)
        d1 = gp1.deriv(params1)
        d2 = gp2.deriv(params2)
        d12 = np.concatenate([d1, d2])
        d = mgp.deriv(params)
        assert_allclose(d, d12)
        h1 = gp1.deriv2(params1)
        h2 = gp2.deriv2(params2)
        h12 = block_diag(h1, h2)
        h = mgp.deriv2(params)
        assert_allclose(h, h12)