import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def make_random_buffer(size, target='host'):
    """Return a host or device buffer with random data.
    """
    if target == 'host':
        assert size >= 0
        buf = pa.allocate_buffer(size)
        assert buf.size == size
        arr = np.frombuffer(buf, dtype=np.uint8)
        assert arr.size == size
        arr[:] = np.random.randint(low=1, high=255, size=size, dtype=np.uint8)
        assert arr.sum() > 0 or size == 0
        arr_ = np.frombuffer(buf, dtype=np.uint8)
        np.testing.assert_equal(arr, arr_)
        return (arr, buf)
    elif target == 'device':
        arr, buf = make_random_buffer(size, target='host')
        dbuf = global_context.new_buffer(size)
        assert dbuf.size == size
        dbuf.copy_from_host(buf, position=0, nbytes=size)
        return (arr, dbuf)
    raise ValueError('invalid target value')