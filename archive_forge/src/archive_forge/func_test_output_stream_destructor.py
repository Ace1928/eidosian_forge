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
def test_output_stream_destructor(tmpdir):
    data = b'some test data\n'
    file_path = tmpdir / 'output_stream.buffered'

    def check_data(file_path, data, **kwargs):
        stream = pa.output_stream(file_path, **kwargs)
        stream.write(data)
        del stream
        gc.collect()
        with open(str(file_path), 'rb') as f:
            return f.read()
    assert check_data(file_path, data, buffer_size=0) == data
    assert check_data(file_path, data, buffer_size=1024) == data