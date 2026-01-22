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
def test_python_file_read():
    data = b'some sample data'
    buf = BytesIO(data)
    f = pa.PythonFile(buf, mode='r')
    assert f.size() == len(data)
    assert f.tell() == 0
    assert f.read(4) == b'some'
    assert f.tell() == 4
    f.seek(0)
    assert f.tell() == 0
    f.seek(5)
    assert f.tell() == 5
    v = f.read(50)
    assert v == b'sample data'
    assert len(v) == 11
    assert f.size() == len(data)
    assert not f.closed
    f.close()
    assert f.closed
    with pytest.raises(TypeError, match='binary file expected'):
        pa.PythonFile(StringIO(), mode='r')