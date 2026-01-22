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
def test_gam_gradient():
    np.random.seed(1)
    pol, y = polynomial_sample_data()
    alpha = 1
    smoother = pol.smoothers[0]
    gp = UnivariateGamPenalty(alpha=alpha, univariate_smoother=smoother)
    for _ in range(10):
        params = np.random.uniform(-2, 2, 4)
        params = np.array([1, 1, 1, 1])
        gam_grad = gp.deriv(params)
        grd = grad(params)
        assert_allclose(gam_grad, grd, rtol=0.01, atol=0.01)