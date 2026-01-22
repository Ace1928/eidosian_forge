import os
import pytest
import sys
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.linalg._svdp import _svdp
from scipy.sparse import csr_matrix, csc_matrix
@pytest.mark.parametrize('shifts', (None, -10, 0, 1, 10, 70))
@pytest.mark.parametrize('dtype', _dtypes[:2])
def test_shifts(shifts, dtype):
    np.random.seed(0)
    n, k = (70, 10)
    A = np.random.random((n, n))
    if shifts is not None and (shifts < 0 or k > min(n - 1 - shifts, n)):
        with pytest.raises(ValueError):
            _svdp(A, k, shifts=shifts, kmax=5 * k, irl_mode=True)
    else:
        _svdp(A, k, shifts=shifts, kmax=5 * k, irl_mode=True)