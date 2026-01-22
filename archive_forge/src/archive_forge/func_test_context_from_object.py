import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
@pytest.mark.parametrize('size', [0, 1, 1000])
def test_context_from_object(size):
    ctx = global_context
    arr, cbuf = make_random_buffer(size, target='device')
    dtype = arr.dtype
    hbuf = cuda.new_host_buffer(size * arr.dtype.itemsize)
    np.frombuffer(hbuf, dtype=dtype)[:] = arr
    cbuf2 = ctx.buffer_from_object(hbuf)
    assert cbuf2.size == cbuf.size
    arr2 = np.frombuffer(cbuf2.copy_to_host(), dtype=dtype)
    np.testing.assert_equal(arr, arr2)
    cbuf2 = ctx.buffer_from_object(cbuf2)
    assert cbuf2.size == cbuf.size
    arr2 = np.frombuffer(cbuf2.copy_to_host(), dtype=dtype)
    np.testing.assert_equal(arr, arr2)
    with pytest.raises(pa.ArrowTypeError, match='buffer is not backed by a CudaBuffer'):
        ctx.buffer_from_object(pa.py_buffer(b'123'))
    with pytest.raises(pa.ArrowTypeError, match="cannot create device buffer view from .* 'numpy.ndarray'"):
        ctx.buffer_from_object(np.array([1, 2, 3]))