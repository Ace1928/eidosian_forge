import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
@pytest.mark.parametrize('size', [0, 1, 1000])
def test_manage_allocate_free_host(size):
    buf = cuda.new_host_buffer(size)
    arr = np.frombuffer(buf, dtype=np.uint8)
    arr[size // 4:3 * size // 4] = 1
    arr_cp = arr.copy()
    arr2 = np.frombuffer(buf, dtype=np.uint8)
    np.testing.assert_equal(arr2, arr_cp)
    assert buf.size == size