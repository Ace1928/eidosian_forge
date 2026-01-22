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
def test_gam_discrete():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, 'results', 'prediction_from_mgcv.csv')
    data_from_r = pd.read_csv(file_path)
    x = data_from_r.x.values
    y = data_from_r.ybin.values
    df = [10]
    degree = [5]
    bsplines = BSplines(x, degree=degree, df=df, include_intercept=True)
    y_mgcv = data_from_r.ybin_est
    alpha = 2e-05
    lg_gam = LogitGam(y, bsplines, alpha=alpha)
    res_lg_gam = lg_gam.fit(maxiter=10000)
    y_gam = np.dot(bsplines.basis, res_lg_gam.params)
    y_gam = sigmoid(y_gam)
    y_mgcv = sigmoid(y_mgcv)
    assert_allclose(y_gam, y_mgcv, rtol=1e-10, atol=0.1)