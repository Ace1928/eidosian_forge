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
def test_get_sqrt():
    n = 1000
    np.random.seed(1)
    x = np.random.normal(0, 1, (n, 3))
    x2 = np.dot(x.T, x)
    sqrt_x2 = matrix_sqrt(x2)
    x2_reconstruction = np.dot(sqrt_x2.T, sqrt_x2)
    assert_allclose(x2_reconstruction, x2)