import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_vector_scalar_abs_step(self):
    x0 = np.array([100.0, -0.5])
    jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0, method='2-point', abs_step=1.49e-08)
    jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0, abs_step=1.49e-08, rel_step=np.inf)
    jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0, method='cs', abs_step=1.49e-08)
    jac_true = self.jac_vector_scalar(x0)
    assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
    assert_allclose(jac_diff_3, jac_true, rtol=3e-09)
    assert_allclose(jac_diff_4, jac_true, rtol=1e-12)