import os
import sys
import pytest
import warnings
import weakref
import numpy as np
import pyarrow as pa
def test_tensor_equals():

    def eq(a, b):
        assert a.equals(b)
        assert a == b
        assert not a != b

    def ne(a, b):
        assert not a.equals(b)
        assert not a == b
        assert a != b
    data = np.random.randn(10, 6, 4)[:, ::2, :]
    tensor1 = pa.Tensor.from_numpy(data)
    tensor2 = pa.Tensor.from_numpy(np.ascontiguousarray(data))
    eq(tensor1, tensor2)
    data = data.copy()
    data[9, 0, 0] = 1.0
    tensor2 = pa.Tensor.from_numpy(np.ascontiguousarray(data))
    ne(tensor1, tensor2)