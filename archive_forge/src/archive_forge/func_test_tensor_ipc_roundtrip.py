import os
import sys
import pytest
import warnings
import weakref
import numpy as np
import pyarrow as pa
def test_tensor_ipc_roundtrip(tmpdir):
    data = np.random.randn(10, 4)
    tensor = pa.Tensor.from_numpy(data)
    path = os.path.join(str(tmpdir), 'pyarrow-tensor-ipc-roundtrip')
    mmap = pa.create_memory_map(path, 1024)
    pa.ipc.write_tensor(tensor, mmap)
    mmap.seek(0)
    result = pa.ipc.read_tensor(mmap)
    assert result.equals(tensor)