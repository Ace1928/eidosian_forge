import os
import sys
import pytest
import warnings
import weakref
import numpy as np
import pyarrow as pa
def test_tensor_size():
    data = np.random.randn(10, 4)
    tensor = pa.Tensor.from_numpy(data)
    assert pa.ipc.get_tensor_size(tensor) > data.size * 8