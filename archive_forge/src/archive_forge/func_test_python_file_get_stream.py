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
@pytest.mark.parametrize('nbytes', (-1, 0, 1, 5, 100))
@pytest.mark.parametrize('file_offset', (-1, 0, 5, 100))
def test_python_file_get_stream(nbytes, file_offset):
    data = b'data1data2data3data4data5'
    f = pa.PythonFile(BytesIO(data), mode='r')
    if nbytes < 0 or file_offset < 0:
        with pytest.raises(pa.ArrowInvalid, match='should be a positive value'):
            f.get_stream(file_offset=file_offset, nbytes=nbytes)
        f.close()
        return
    else:
        stream = f.get_stream(file_offset=file_offset, nbytes=nbytes)
    start = min(file_offset, len(data))
    end = min(file_offset + nbytes, len(data))
    buf = BytesIO(data[start:end])
    assert stream.read(nbytes=4) == buf.read(4)
    assert stream.read(nbytes=6) == buf.read(6)
    assert stream.read() == buf.read()
    n = len(data) * 2
    assert stream.read(n) == buf.read(n)
    with pytest.raises(OSError, match='seekable'):
        stream.seek(0)
    stream.close()
    assert stream.closed