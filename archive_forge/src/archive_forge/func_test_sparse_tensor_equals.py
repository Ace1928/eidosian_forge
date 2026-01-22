import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
@pytest.mark.parametrize('sparse_tensor_type', [pa.SparseCSRMatrix, pa.SparseCSCMatrix, pa.SparseCOOTensor, pa.SparseCSFTensor])
def test_sparse_tensor_equals(sparse_tensor_type):

    def eq(a, b):
        assert a.equals(b)
        assert a == b
        assert not a != b

    def ne(a, b):
        assert not a.equals(b)
        assert not a == b
        assert a != b
    data = np.random.randn(10, 6)[:, ::2]
    sparse_tensor1 = sparse_tensor_type.from_dense_numpy(data)
    sparse_tensor2 = sparse_tensor_type.from_dense_numpy(np.ascontiguousarray(data))
    eq(sparse_tensor1, sparse_tensor2)
    data = data.copy()
    data[9, 0] = 1.0
    sparse_tensor2 = sparse_tensor_type.from_dense_numpy(np.ascontiguousarray(data))
    ne(sparse_tensor1, sparse_tensor2)