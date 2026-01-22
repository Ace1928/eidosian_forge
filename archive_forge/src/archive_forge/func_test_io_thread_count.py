import os
import subprocess
import sys
import pytest
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
def test_io_thread_count():
    n = pa.io_thread_count()
    assert n > 0
    try:
        pa.set_io_thread_count(n + 5)
        assert pa.io_thread_count() == n + 5
    finally:
        pa.set_io_thread_count(n)