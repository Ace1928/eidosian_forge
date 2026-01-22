import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_csr_matrix_from_dense(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([[8, 2, 5, 3, 4, 6]]).T.astype(dtype)
    indptr = np.array([0, 2, 3, 4, 6])
    indices = np.array([0, 2, 5, 0, 4, 5])
    array = np.array([[8, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 5], [3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 6]]).astype(dtype)
    tensor = pa.Tensor.from_numpy(array)
    sparse_tensor = pa.SparseCSRMatrix.from_dense_numpy(array)
    repr(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr, result_indptr)
    assert np.array_equal(indices, result_indices)
    sparse_tensor = pa.SparseCSRMatrix.from_tensor(tensor)
    repr(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr, result_indptr)
    assert np.array_equal(indices, result_indices)