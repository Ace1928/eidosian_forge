import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def test_default_allocated_bytes():
    pool = pa.default_memory_pool()
    with allocate_bytes(pool, 1024):
        check_allocated_bytes(pool)
        assert pool.bytes_allocated() == pa.total_allocated_bytes()