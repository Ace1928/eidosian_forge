from __future__ import annotations
import pytest
from dask.hashing import hash_buffer, hash_buffer_hex, hashers
@pytest.mark.parametrize('x', buffers)
def test_hash_buffer_hex(x):
    for hasher in [None] + hashers:
        h = hash_buffer_hex(x, hasher=hasher)
        assert isinstance(h, str)
        assert 16 <= len(h) < 64
        assert h == hash_buffer_hex(x, hasher=hasher)