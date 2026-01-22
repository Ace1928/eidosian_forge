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
def test_type_conversion_mixed_sign_or_complex_spectrum(self):
    complex_dtype_chars = ('F', 'D', 'G')
    for matrix_as_list in ([[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]]):
        W = scipy.linalg.eigvals(matrix_as_list)
        assert_(any((w.imag or w.real < 0 for w in W)))
        for p in (-2.4, -0.9, 0.2, 3.3):
            A = np.array(matrix_as_list, dtype=complex)
            A_power = fractional_matrix_power(A, p)
            assert_(A_power.dtype.char in complex_dtype_chars)
            A = np.array(matrix_as_list, dtype=float)
            A_power = fractional_matrix_power(A, p)
            assert_(A_power.dtype.char in complex_dtype_chars)