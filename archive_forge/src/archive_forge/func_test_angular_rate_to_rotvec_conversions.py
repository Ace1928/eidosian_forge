from itertools import product
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.spatial.transform._rotation_spline import (
def test_angular_rate_to_rotvec_conversions():
    np.random.seed(0)
    rv = np.random.randn(4, 3)
    A = _angular_rate_to_rotvec_dot_matrix(rv)
    A_inv = _rotvec_dot_to_angular_rate_matrix(rv)
    assert_allclose(_matrix_vector_product_of_stacks(A, rv), rv)
    assert_allclose(_matrix_vector_product_of_stacks(A_inv, rv), rv)
    I_stack = np.empty((4, 3, 3))
    I_stack[:] = np.eye(3)
    assert_allclose(np.matmul(A, A_inv), I_stack, atol=1e-15)