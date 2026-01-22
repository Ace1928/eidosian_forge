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
def test_data_size_preservation_float_in_float_out(self):
    M = np.zeros((10, 10), dtype=np.float16)
    assert sqrtm(M).dtype == np.float16
    M = np.zeros((10, 10), dtype=np.float32)
    assert sqrtm(M).dtype == np.float32
    M = np.zeros((10, 10), dtype=np.float64)
    assert sqrtm(M).dtype == np.float64
    if hasattr(np, 'float128'):
        M = np.zeros((10, 10), dtype=np.float128)
        assert sqrtm(M).dtype == np.float128