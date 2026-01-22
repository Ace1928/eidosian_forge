import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def test_default_backend_name():
    pool = pa.default_memory_pool()
    assert pool.backend_name in possible_backends