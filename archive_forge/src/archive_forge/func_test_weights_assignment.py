import numpy as np
from numpy.testing import assert_allclose, assert_equal
import statsmodels.base._penalties as smpen
from statsmodels.tools.numdiff import approx_fprime, approx_hess
def test_weights_assignment(self):
    weights = [1.0, 2.0]
    pen = smpen.L2(weights=weights)
    assert_equal(pen.weights, weights)