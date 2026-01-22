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
def test_multivariate_cubic_splines():
    np.random.seed(0)
    from statsmodels.gam.smooth_basis import CubicSplines
    n = 500
    x1 = np.linspace(-3, 3, n)
    x2 = np.linspace(0, 1, n) ** 2
    x = np.vstack([x1, x2]).T
    y1 = np.sin(x1) / x1
    y2 = x2 * x2
    y0 = y1 + y2
    y = y0 + np.random.normal(0, 0.3 / 2, n)
    y -= y.mean()
    y0 -= y0.mean()
    alphas = [0.001, 0.001]
    cs = CubicSplines(x, df=[10, 10], constraints='center')
    gam = GLMGam(y, exog=np.ones((n, 1)), smoother=cs, alpha=alphas)
    gam_res = gam.fit(method='pirls')
    y_est = gam_res.fittedvalues
    y_est -= y_est.mean()
    index = list(range(50, n - 50))
    y_est = y_est[index]
    y0 = y0[index]
    y = y[index]
    assert_allclose(y_est, y0, atol=0.04)