import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_coo_tensor_from_dense(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    expected_data = np.array([[8, 2, 5, 3, 4, 6]]).T.astype(dtype)
    expected_coords = np.array([[0, 0, 1, 2, 3, 3], [0, 2, 5, 0, 4, 5]]).T
    array = np.array([[8, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 5], [3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 6]]).astype(dtype)
    tensor = pa.Tensor.from_numpy(array)
    sparse_tensor = pa.SparseCOOTensor.from_dense_numpy(array)
    repr(sparse_tensor)
    result_data, result_coords = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(expected_data, result_data)
    assert np.array_equal(expected_coords, result_coords)
    sparse_tensor = pa.SparseCOOTensor.from_tensor(tensor)
    repr(sparse_tensor)
    result_data, result_coords = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(expected_data, result_data)
    assert np.array_equal(expected_coords, result_coords)