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
def polynomial_sample_data():
    """A polynomial of degree 4

    poly = ax^4 + bx^3 + cx^2 + dx + e
    second der = 12ax^2 + 6bx + 2c
    integral from -1 to 1 of second der^2 is
        (288 a^2)/5 + 32 a c + 8 (3 b^2 + c^2)
    the gradient of the integral is der
        [576*a/5 + 32 * c, 48*b, 32*a + 16*c, 0, 0]

    Returns
    -------
    poly : smoother instance
    y : ndarray
        generated function values, demeaned
    """
    n = 10000
    x = np.linspace(-1, 1, n)
    y = 2 * x ** 3 - x
    y -= y.mean()
    degree = [4]
    pol = PolynomialSmoother(x, degree)
    return (pol, y)