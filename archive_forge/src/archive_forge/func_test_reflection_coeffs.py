import numpy as np
from scipy.linalg._solve_toeplitz import levinson
from scipy.linalg import solve, toeplitz, solve_toeplitz
from numpy.testing import assert_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
def test_reflection_coeffs():
    random = np.random.RandomState(1234)
    y_d = random.randn(10)
    y_z = random.randn(10) + 1j
    reflection_coeffs_d = [1]
    reflection_coeffs_z = [1]
    for i in range(2, 10):
        reflection_coeffs_d.append(solve_toeplitz(y_d[:i - 1], b=y_d[1:i])[-1])
        reflection_coeffs_z.append(solve_toeplitz(y_z[:i - 1], b=y_z[1:i])[-1])
    y_d_concat = np.concatenate((y_d[-2:0:-1], y_d[:-1]))
    y_z_concat = np.concatenate((y_z[-2:0:-1].conj(), y_z[:-1]))
    _, ref_d = levinson(y_d_concat, b=y_d[1:])
    _, ref_z = levinson(y_z_concat, b=y_z[1:])
    assert_allclose(reflection_coeffs_d, ref_d[:-1])
    assert_allclose(reflection_coeffs_z, ref_z[:-1])