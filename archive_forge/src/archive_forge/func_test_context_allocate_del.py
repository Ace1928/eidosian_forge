import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def test_context_allocate_del():
    bytes_allocated = global_context.bytes_allocated
    cudabuf = global_context.new_buffer(128)
    assert global_context.bytes_allocated == bytes_allocated + 128
    del cudabuf
    assert global_context.bytes_allocated == bytes_allocated