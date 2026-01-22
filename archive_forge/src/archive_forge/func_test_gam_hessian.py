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
def test_gam_hessian():
    np.random.seed(1)
    pol, y = polynomial_sample_data()
    univ_pol = pol.smoothers[0]
    alpha = 1
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=univ_pol)
    for _ in range(10):
        params = np.random.randint(-2, 2, 5)
        gam_der2 = gp.deriv2(params)
        hess = hessian(params)
        hess = np.flipud(hess)
        hess = np.fliplr(hess)
        assert_allclose(gam_der2, hess, atol=1e-13, rtol=0.001)