import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
@pytest.mark.parametrize('size', [0, 1, 1000])
def test_context_device_buffer(size):
    arr, buf = make_random_buffer(size)
    cudabuf = global_context.buffer_from_data(buf)
    assert cudabuf.size == size
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)
    with pytest.raises(BufferError):
        memoryview(cudabuf)
    cudabuf = global_context.buffer_from_data(arr)
    assert cudabuf.size == size
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)
    cudabuf = global_context.buffer_from_data(arr.tobytes())
    assert cudabuf.size == size
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)
    cudabuf2 = cudabuf.slice(0, cudabuf.size)
    assert cudabuf2.size == size
    arr2 = np.frombuffer(cudabuf2.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)
    if size > 1:
        cudabuf2.copy_from_host(arr[size // 2:])
        arr3 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
        np.testing.assert_equal(np.concatenate((arr[size // 2:], arr[size // 2:])), arr3)
        cudabuf2.copy_from_host(arr[:size // 2])
    cudabuf2 = global_context.buffer_from_data(cudabuf)
    assert cudabuf2.size == size
    arr2 = np.frombuffer(cudabuf2.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)
    cudabuf2.copy_from_host(arr[size // 2:])
    arr3 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr, arr3)
    cudabuf2 = cudabuf.slice(0, cudabuf.size + 10)
    assert cudabuf2.size == size
    arr2 = np.frombuffer(cudabuf2.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)
    cudabuf2 = cudabuf.slice(size // 4, size + 10)
    assert cudabuf2.size == size - size // 4
    arr2 = np.frombuffer(cudabuf2.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr[size // 4:], arr2)
    soffset = size // 4
    ssize = 2 * size // 4
    cudabuf = global_context.buffer_from_data(buf, offset=soffset, size=ssize)
    assert cudabuf.size == ssize
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr[soffset:soffset + ssize], arr2)
    cudabuf = global_context.buffer_from_data(buf.slice(offset=soffset, length=ssize))
    assert cudabuf.size == ssize
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr[soffset:soffset + ssize], arr2)
    cudabuf = global_context.buffer_from_data(arr, offset=soffset, size=ssize)
    assert cudabuf.size == ssize
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr[soffset:soffset + ssize], arr2)
    cudabuf = global_context.buffer_from_data(arr[soffset:soffset + ssize])
    assert cudabuf.size == ssize
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr[soffset:soffset + ssize], arr2)
    cudabuf = global_context.buffer_from_data(arr.tobytes(), offset=soffset, size=ssize)
    assert cudabuf.size == ssize
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr[soffset:soffset + ssize], arr2)
    cudabuf = global_context.new_buffer(size)
    assert cudabuf.size == size
    cudabuf = global_context.buffer_from_data(arr)
    cudabuf2 = cudabuf.slice(soffset, ssize)
    assert cudabuf2.size == ssize
    arr2 = np.frombuffer(cudabuf2.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr[soffset:soffset + ssize], arr2)
    buf = cuda.new_host_buffer(size)
    arr_ = np.frombuffer(buf, dtype=np.uint8)
    arr_[:] = arr
    cudabuf = global_context.buffer_from_data(buf)
    assert cudabuf.size == size
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr, arr2)
    cudabuf = global_context.buffer_from_data(buf, offset=soffset, size=ssize)
    assert cudabuf.size == ssize
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr[soffset:soffset + ssize], arr2)
    cudabuf = global_context.buffer_from_data(buf.slice(offset=soffset, length=ssize))
    assert cudabuf.size == ssize
    arr2 = np.frombuffer(cudabuf.copy_to_host(), dtype=np.uint8)
    np.testing.assert_equal(arr[soffset:soffset + ssize], arr2)