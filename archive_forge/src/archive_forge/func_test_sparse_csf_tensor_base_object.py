import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
def test_sparse_csf_tensor_base_object():
    data = np.array([[8, 2, 5, 3, 4, 6]]).T
    indptr = [np.array([0, 2, 3, 4, 6])]
    indices = [np.array([0, 1, 2, 3]), np.array([0, 2, 5, 0, 4, 5])]
    array = np.array([[8, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 5], [3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 6]])
    sparse_tensor = pa.SparseCSFTensor.from_dense_numpy(array)
    n = sys.getrefcount(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sys.getrefcount(sparse_tensor) == n + 4
    sparse_tensor = None
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr[0], result_indptr[0])
    assert np.array_equal(indices[0], result_indices[0])
    assert np.array_equal(indices[1], result_indices[1])