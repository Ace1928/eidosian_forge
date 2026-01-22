import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def test_writes(total_size, chunksize, buffer_size=0):
    cbuf, writer = allocate(total_size)
    arr, buf = make_random_buffer(size=total_size, target='host')
    if buffer_size > 0:
        writer.buffer_size = buffer_size
    position = writer.tell()
    assert position == 0
    writer.write(buf.slice(length=chunksize))
    assert writer.tell() == chunksize
    writer.seek(0)
    position = writer.tell()
    assert position == 0
    while position < total_size:
        bytes_to_write = min(chunksize, total_size - position)
        writer.write(buf.slice(offset=position, length=bytes_to_write))
        position += bytes_to_write
    writer.flush()
    assert cbuf.size == total_size
    cbuf.context.synchronize()
    buf2 = cbuf.copy_to_host()
    cbuf.context.synchronize()
    assert buf2.size == total_size
    arr2 = np.frombuffer(buf2, dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)