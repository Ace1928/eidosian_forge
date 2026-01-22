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
def test_multivariate_gam_cv_path():

    def sample_metric(y1, y2):
        return np.linalg.norm(y1 - y2) / len(y1)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, 'results', 'prediction_from_mgcv.csv')
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.y.values
    se_from_mgcv = data_from_r.y_est_se
    y_mgcv = data_from_r.y_mgcv_gcv
    df = [10]
    degree = [6]
    bsplines = BSplines(x, degree=degree, df=df, include_intercept=True)
    gam = GLMGam
    alphas = [np.linspace(0, 2, 10)]
    k = 3
    cv = KFold(k_folds=k, shuffle=True)
    np.random.seed(123)
    gam_cv = MultivariateGAMCVPath(smoother=bsplines, alphas=alphas, gam=gam, cost=sample_metric, endog=y, exog=None, cv_iterator=cv)
    gam_cv_res = gam_cv.fit()
    glm_gam = GLMGam(y, smoother=bsplines, alpha=gam_cv.alpha_cv)
    res_glm_gam = glm_gam.fit(method='irls', max_start_irls=0, disp=1, maxiter=10000)
    y_est = res_glm_gam.predict(bsplines.basis)
    assert_allclose(data_from_r.y_mgcv_gcv, y_est, atol=0.1, rtol=0.1)
    np.random.seed(123)
    alpha_cv, res_cv = glm_gam.select_penweight_kfold(alphas=alphas, k_folds=3)
    assert_allclose(alpha_cv, gam_cv.alpha_cv, rtol=1e-12)