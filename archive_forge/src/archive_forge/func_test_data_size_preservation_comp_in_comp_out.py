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
def test_data_size_preservation_comp_in_comp_out(self):
    M = np.array([[2j, 4], [0, -2j]], dtype=np.complex64)
    assert sqrtm(M).dtype == np.complex128
    if hasattr(np, 'complex256'):
        M = np.array([[2j, 4], [0, -2j]], dtype=np.complex128)
        assert sqrtm(M).dtype == np.complex256
        M = np.array([[2j, 4], [0, -2j]], dtype=np.complex256)
        assert sqrtm(M).dtype == np.complex256