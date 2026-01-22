import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def check_allocated_bytes(pool):
    """
    Check allocation stats on *pool*.
    """
    allocated_before = pool.bytes_allocated()
    max_mem_before = pool.max_memory()
    with allocate_bytes(pool, 512):
        assert pool.bytes_allocated() == allocated_before + 512
        new_max_memory = pool.max_memory()
        assert pool.max_memory() >= max_mem_before
    assert pool.bytes_allocated() == allocated_before
    assert pool.max_memory() == new_max_memory