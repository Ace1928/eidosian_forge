from itertools import product
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.spatial.transform._rotation_spline import (
def test_create_block_3_diagonal_matrix():
    np.random.seed(0)
    A = np.empty((4, 3, 3))
    A[:] = np.arange(1, 5)[:, None, None]
    B = np.empty((4, 3, 3))
    B[:] = -np.arange(1, 5)[:, None, None]
    d = 10 * np.arange(10, 15)
    banded = _create_block_3_diagonal_matrix(A, B, d)
    k, l = list(zip(*product(np.arange(banded.shape[0]), np.arange(banded.shape[1]))))
    k = np.asarray(k)
    l = np.asarray(l)
    i = k - 5 + l
    j = l
    values = banded.ravel()
    mask = (i >= 0) & (i < 15)
    i = i[mask]
    j = j[mask]
    values = values[mask]
    full = np.zeros((15, 15))
    full[i, j] = values
    zero = np.zeros((3, 3))
    eye = np.eye(3)
    ref = np.block([[d[0] * eye, B[0], zero, zero, zero], [A[0], d[1] * eye, B[1], zero, zero], [zero, A[1], d[2] * eye, B[2], zero], [zero, zero, A[2], d[3] * eye, B[3]], [zero, zero, zero, A[3], d[4] * eye]])
    assert_allclose(full, ref, atol=1e-19)