import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pytest
import sys
import tempfile
import weakref
import numpy as np
from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa
def test_buffer_slicing():
    data = b'some data!'
    buf = pa.py_buffer(data)
    sliced = buf.slice(2)
    expected = pa.py_buffer(b'me data!')
    assert sliced.equals(expected)
    sliced2 = buf.slice(2, 4)
    expected2 = pa.py_buffer(b'me d')
    assert sliced2.equals(expected2)
    assert buf.slice(0).equals(buf)
    assert len(buf.slice(len(buf))) == 0
    with pytest.raises(IndexError):
        buf.slice(-1)
    with pytest.raises(IndexError):
        buf.slice(len(buf) + 1)
    assert buf[11:].to_pybytes() == b''
    with pytest.raises(IndexError):
        buf.slice(1, len(buf))
    assert buf[1:11].to_pybytes() == buf.to_pybytes()[1:]
    with pytest.raises(IndexError):
        buf.slice(1, -1)
    assert buf[2:].equals(buf.slice(2))
    assert buf[2:5].equals(buf.slice(2, 3))
    assert buf[-5:].equals(buf.slice(len(buf) - 5))
    assert buf[-5:-2].equals(buf.slice(len(buf) - 5, 3))
    with pytest.raises(IndexError):
        buf[::-1]
    with pytest.raises(IndexError):
        buf[::2]
    n = len(buf)
    for start in range(-n * 2, n * 2):
        for stop in range(-n * 2, n * 2):
            assert buf[start:stop].to_pybytes() == buf.to_pybytes()[start:stop]