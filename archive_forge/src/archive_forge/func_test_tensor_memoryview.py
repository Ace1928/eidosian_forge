import os
import sys
import pytest
import warnings
import weakref
import numpy as np
import pyarrow as pa
def test_tensor_memoryview():
    for dtype, expected_format in [(np.int8, '=b'), (np.int64, '=q'), (np.uint64, '=Q'), (np.float16, 'e'), (np.float64, 'd')]:
        data = np.arange(10, dtype=dtype)
        dtype = data.dtype
        lst = data.tolist()
        tensor = pa.Tensor.from_numpy(data)
        m = memoryview(tensor)
        assert m.format == expected_format
        assert m.shape == data.shape
        assert m.strides == data.strides
        assert m.ndim == 1
        assert m.nbytes == data.nbytes
        assert m.itemsize == data.itemsize
        assert m.itemsize * 8 == tensor.type.bit_width
        assert np.frombuffer(m, dtype).tolist() == lst
        del tensor, data
        assert np.frombuffer(m, dtype).tolist() == lst