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
@pytest.mark.gzip
def test_input_stream_python_file(tmpdir):
    data = b'some test data\n' * 10 + b'eof\n'
    bio = BytesIO(data)
    stream = pa.input_stream(bio)
    assert stream.read() == data
    gz_data = gzip.compress(data)
    bio = BytesIO(gz_data)
    stream = pa.input_stream(bio)
    assert stream.read() == gz_data
    bio.seek(0)
    stream = pa.input_stream(bio, compression='gzip')
    assert stream.read() == data
    file_path = tmpdir / 'input_stream'
    with open(str(file_path), 'wb') as f:
        f.write(data)
    with open(str(file_path), 'rb') as f:
        stream = pa.input_stream(f)
        assert stream.read() == data