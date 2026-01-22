import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse
from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('arr_type', [np.array, sparse.csr_matrix, sparse.coo_matrix, sparse.csr_array, sparse.coo_array])
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('use_out_degree', [True, False])
def test_asymmetric_laplacian(use_out_degree, normed, copy, dtype, arr_type):
    A = [[0, 1, 0], [4, 2, 0], [0, 0, 0]]
    A = arr_type(np.array(A), dtype=dtype)
    A_copy = A.copy()
    if not normed and use_out_degree:
        L = [[1, -1, 0], [-4, 4, 0], [0, 0, 0]]
        d = [1, 4, 0]
    if normed and use_out_degree:
        L = [[1, -0.5, 0], [-2, 1, 0], [0, 0, 0]]
        d = [1, 2, 1]
    if not normed and (not use_out_degree):
        L = [[4, -1, 0], [-4, 1, 0], [0, 0, 0]]
        d = [4, 1, 0]
    if normed and (not use_out_degree):
        L = [[1, -0.5, 0], [-2, 1, 0], [0, 0, 0]]
        d = [2, 1, 1]
    _check_laplacian_dtype_none(A, L, d, normed=normed, use_out_degree=use_out_degree, copy=copy, dtype=dtype, arr_type=arr_type)
    _check_laplacian_dtype(A_copy, L, d, normed=normed, use_out_degree=use_out_degree, copy=copy, dtype=dtype, arr_type=arr_type)