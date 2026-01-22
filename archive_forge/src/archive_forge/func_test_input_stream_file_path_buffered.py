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
def test_input_stream_file_path_buffered(tmpdir):
    data = b'some test data\n' * 10 + b'eof\n'
    file_path = tmpdir / 'input_stream.buffered'
    with open(str(file_path), 'wb') as f:
        f.write(data)
    stream = pa.input_stream(file_path, buffer_size=32)
    assert isinstance(stream, pa.BufferedInputStream)
    assert stream.read() == data
    stream = pa.input_stream(str(file_path), buffer_size=64)
    assert isinstance(stream, pa.BufferedInputStream)
    assert stream.read() == data
    stream = pa.input_stream(pathlib.Path(str(file_path)), buffer_size=1024)
    assert isinstance(stream, pa.BufferedInputStream)
    assert stream.read() == data
    unbuffered_stream = pa.input_stream(file_path, buffer_size=0)
    assert isinstance(unbuffered_stream, pa.OSFile)
    msg = 'Buffer size must be larger than zero'
    with pytest.raises(ValueError, match=msg):
        pa.input_stream(file_path, buffer_size=-1)
    with pytest.raises(TypeError):
        pa.input_stream(file_path, buffer_size='million')