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
def test_native_file_modes(tmpdir):
    path = os.path.join(str(tmpdir), guid())
    with open(path, 'wb') as f:
        f.write(b'foooo')
    with pa.OSFile(path, mode='r') as f:
        assert f.mode == 'rb'
        assert f.readable()
        assert not f.writable()
        assert f.seekable()
    with pa.OSFile(path, mode='rb') as f:
        assert f.mode == 'rb'
        assert f.readable()
        assert not f.writable()
        assert f.seekable()
    with pa.OSFile(path, mode='w') as f:
        assert f.mode == 'wb'
        assert not f.readable()
        assert f.writable()
        assert not f.seekable()
    with pa.OSFile(path, mode='wb') as f:
        assert f.mode == 'wb'
        assert not f.readable()
        assert f.writable()
        assert not f.seekable()
    with pa.OSFile(path, mode='ab') as f:
        assert f.mode == 'ab'
        assert not f.readable()
        assert f.writable()
        assert not f.seekable()
    with pa.OSFile(path, mode='a') as f:
        assert f.mode == 'ab'
        assert not f.readable()
        assert f.writable()
        assert not f.seekable()
    with open(path, 'wb') as f:
        f.write(b'foooo')
    with pa.memory_map(path, 'r') as f:
        assert f.mode == 'rb'
        assert f.readable()
        assert not f.writable()
        assert f.seekable()
    with pa.memory_map(path, 'r+') as f:
        assert f.mode == 'rb+'
        assert f.readable()
        assert f.writable()
        assert f.seekable()
    with pa.memory_map(path, 'r+b') as f:
        assert f.mode == 'rb+'
        assert f.readable()
        assert f.writable()
        assert f.seekable()