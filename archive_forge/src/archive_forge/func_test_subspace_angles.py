import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
def test_subspace_angles():
    H = hadamard(8, float)
    A = H[:, :3]
    B = H[:, 3:]
    assert_allclose(subspace_angles(A, B), [np.pi / 2.0] * 3, atol=1e-14)
    assert_allclose(subspace_angles(B, A), [np.pi / 2.0] * 3, atol=1e-14)
    for x in (A, B):
        assert_allclose(subspace_angles(x, x), np.zeros(x.shape[1]), atol=1e-14)
    x = np.array([[0.5376671395461, 0.318928239858981, 3.57839693972576, 0.725404224946106], [1.833885014595086, -1.307688296305273, 2.769437029884877, -0.063054873189656], [-2.258846861003648, -0.433592022305684, -1.349886940156521, 0.714742903826096], [0.862173320368121, 0.34262446653865, 3.034923466331855, -0.204966058299775]])
    expected = 1.481454682101605
    assert_allclose(subspace_angles(x[:, :2], x[:, 2:])[0], expected, rtol=1e-12)
    assert_allclose(subspace_angles(x[:, 2:], x[:, :2])[0], expected, rtol=1e-12)
    expected = 0.746361174247302
    assert_allclose(subspace_angles(x[:, :2], x[:, [2]]), expected, rtol=1e-12)
    assert_allclose(subspace_angles(x[:, [2]], x[:, :2]), expected, rtol=1e-12)
    expected = 0.487163718534313
    assert_allclose(subspace_angles(x[:, :3], x[:, [3]]), expected, rtol=1e-12)
    assert_allclose(subspace_angles(x[:, [3]], x[:, :3]), expected, rtol=1e-12)
    expected = 0.328950515907756
    assert_allclose(subspace_angles(x[:, :2], x[:, 1:]), [expected, 0], atol=1e-12)
    assert_raises(ValueError, subspace_angles, x[0], x)
    assert_raises(ValueError, subspace_angles, x, x[0])
    assert_raises(ValueError, subspace_angles, x[:-1], x)
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
    B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]])
    expected = np.array([np.pi / 2, 0, 0])
    assert_allclose(subspace_angles(A, B), expected, rtol=1e-12)
    a = [[1 + 1j], [0]]
    b = [[1 - 1j, 0], [0, 1]]
    assert_allclose(subspace_angles(a, b), 0.0, atol=1e-14)
    assert_allclose(subspace_angles(b, a), 0.0, atol=1e-14)