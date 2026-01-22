import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def test_BufferReader():
    size = 1000
    arr, cbuf = make_random_buffer(size=size, target='device')
    reader = cuda.BufferReader(cbuf)
    reader.seek(950)
    assert reader.tell() == 950
    data = reader.read(100)
    assert len(data) == 50
    assert reader.tell() == 1000
    reader.seek(925)
    arr2 = np.zeros(100, dtype=np.uint8)
    n = reader.readinto(arr2)
    assert n == 75
    assert reader.tell() == 1000
    np.testing.assert_equal(arr[925:], arr2[:75])
    reader.seek(0)
    assert reader.tell() == 0
    buf2 = reader.read_buffer()
    arr2 = np.frombuffer(buf2.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)