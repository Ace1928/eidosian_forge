import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_non_numpy(self):
    x0 = 1.0
    jac_true = self.jac_non_numpy(x0)
    jac_diff_2 = approx_derivative(self.jac_non_numpy, x0, method='2-point')
    jac_diff_3 = approx_derivative(self.jac_non_numpy, x0)
    assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
    assert_allclose(jac_diff_3, jac_true, rtol=1e-08)
    assert_raises(TypeError, approx_derivative, self.jac_non_numpy, x0, **dict(method='cs'))