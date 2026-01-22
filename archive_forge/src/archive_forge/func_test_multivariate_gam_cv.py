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
def test_multivariate_gam_cv():

    def cost(x1, x2):
        return np.linalg.norm(x1 - x2) / len(x1)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, 'results', 'prediction_from_mgcv.csv')
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.y.values
    df = [10]
    degree = [5]
    bsplines = BSplines(x, degree=degree, df=df)
    alphas = [0.0251]
    alphas = [2]
    cv = KFold(3)
    gp = MultivariateGamPenalty(bsplines, alpha=alphas)
    gam_cv = MultivariateGAMCV(smoother=bsplines, alphas=alphas, gam=GLMGam, cost=cost, endog=y, exog=None, cv_iterator=cv)
    gam_cv_res = gam_cv.fit()