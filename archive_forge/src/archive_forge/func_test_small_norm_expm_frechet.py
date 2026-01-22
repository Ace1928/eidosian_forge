import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
def test_small_norm_expm_frechet(self):
    M_original = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 1, 2], [0, 0, 5, 6]], dtype=float)
    A_original = np.array([[1, 2], [5, 6]], dtype=float)
    E_original = np.array([[3, 4], [7, 8]], dtype=float)
    A_original_norm_1 = scipy.linalg.norm(A_original, 1)
    selected_m_list = [1, 3, 5, 7, 9, 11, 13, 15]
    m_neighbor_pairs = zip(selected_m_list[:-1], selected_m_list[1:])
    for ma, mb in m_neighbor_pairs:
        ell_a = scipy.linalg._expm_frechet.ell_table_61[ma]
        ell_b = scipy.linalg._expm_frechet.ell_table_61[mb]
        target_norm_1 = 0.5 * (ell_a + ell_b)
        scale = target_norm_1 / A_original_norm_1
        M = scale * M_original
        A = scale * A_original
        E = scale * E_original
        expected_expm = scipy.linalg.expm(A)
        expected_frechet = scipy.linalg.expm(M)[:2, 2:]
        observed_expm, observed_frechet = expm_frechet(A, E)
        assert_allclose(expected_expm, observed_expm)
        assert_allclose(expected_frechet, observed_frechet)