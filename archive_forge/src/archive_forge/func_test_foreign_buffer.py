import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def test_foreign_buffer():
    ctx = global_context
    dtype = np.dtype(np.uint8)
    size = 10
    hbuf = cuda.new_host_buffer(size * dtype.itemsize)
    rc = sys.getrefcount(hbuf)
    fbuf = ctx.foreign_buffer(hbuf.address, hbuf.size, hbuf)
    assert sys.getrefcount(hbuf) == rc + 1
    del fbuf
    assert sys.getrefcount(hbuf) == rc
    fbuf = ctx.foreign_buffer(hbuf.address, hbuf.size, hbuf)
    del hbuf
    fbuf.copy_to_host()
    hbuf = cuda.new_host_buffer(size * dtype.itemsize)
    fbuf = ctx.foreign_buffer(hbuf.address, hbuf.size)
    del hbuf
    with pytest.raises(pa.ArrowIOError, match='Cuda error '):
        fbuf.copy_to_host()