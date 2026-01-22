import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
def test_sparse_coo_tensor_base_object():
    expected_data = np.array([[8, 2, 5, 3, 4, 6]]).T
    expected_coords = np.array([[0, 0, 1, 2, 3, 3], [0, 2, 5, 0, 4, 5]]).T
    array = np.array([[8, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 5], [3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 6]])
    sparse_tensor = pa.SparseCOOTensor.from_dense_numpy(array)
    n = sys.getrefcount(sparse_tensor)
    result_data, result_coords = sparse_tensor.to_numpy()
    assert sparse_tensor.has_canonical_format
    assert sys.getrefcount(sparse_tensor) == n + 2
    sparse_tensor = None
    assert np.array_equal(expected_data, result_data)
    assert np.array_equal(expected_coords, result_coords)
    assert result_coords.flags.c_contiguous