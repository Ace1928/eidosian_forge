import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def test_supported_memory_backends():
    backends = pa.supported_memory_backends()
    assert 'system' in backends
    if should_have_jemalloc:
        assert 'jemalloc' in backends
    if should_have_mimalloc:
        assert 'mimalloc' in backends