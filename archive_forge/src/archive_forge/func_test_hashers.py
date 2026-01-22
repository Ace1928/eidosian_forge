from __future__ import annotations
import pytest
from dask.hashing import hash_buffer, hash_buffer_hex, hashers
@pytest.mark.parametrize('hasher', hashers)
def test_hashers(hasher):
    x = b'x'
    h = hasher(x)
    assert isinstance(h, bytes)
    assert 8 <= len(h) < 32