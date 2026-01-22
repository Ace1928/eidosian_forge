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
def test_input_stream_buffer():
    data = b'some test data\n' * 10 + b'eof\n'
    for arg in [pa.py_buffer(data), memoryview(data)]:
        stream = pa.input_stream(arg)
        assert stream.read() == data
    gz_data = gzip.compress(data)
    stream = pa.input_stream(memoryview(gz_data))
    assert stream.read() == gz_data
    stream = pa.input_stream(memoryview(gz_data), compression='gzip')
    assert stream.read() == data