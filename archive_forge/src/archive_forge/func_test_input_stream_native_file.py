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
def test_input_stream_native_file():
    data = b'some test data\n' * 10 + b'eof\n'
    gz_data = gzip.compress(data)
    reader = pa.BufferReader(gz_data)
    stream = pa.input_stream(reader)
    assert stream is reader
    reader = pa.BufferReader(gz_data)
    stream = pa.input_stream(reader, compression='gzip')
    assert stream.read() == data