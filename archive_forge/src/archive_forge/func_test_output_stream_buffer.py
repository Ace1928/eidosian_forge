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
def test_output_stream_buffer():
    data = b'some test data\n' * 10 + b'eof\n'
    buf = bytearray(len(data))
    stream = pa.output_stream(pa.py_buffer(buf))
    stream.write(data)
    assert buf == data
    buf = bytearray(len(data))
    stream = pa.output_stream(memoryview(buf))
    stream.write(data)
    assert buf == data