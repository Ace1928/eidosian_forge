from __future__ import annotations
import io
import os
import pathlib
import pytest
from fsspec.utils import (
def test_seek_delimiter_endline():
    f = io.BytesIO(b'123\n456\n789')
    seek_delimiter(f, b'\n', 5)
    assert f.tell() == 0
    for bs in [1, 5, 100]:
        f.seek(1)
        seek_delimiter(f, b'\n', blocksize=bs)
        assert f.tell() == 4
    f = io.BytesIO(b'123abc456abc789')
    for bs in [1, 2, 3, 4, 5, 6, 10]:
        f.seek(1)
        seek_delimiter(f, b'abc', blocksize=bs)
        assert f.tell() == 6
    f = io.BytesIO(b'123\n456')
    f.seek(5)
    seek_delimiter(f, b'\n', 5)
    assert f.tell() == 7