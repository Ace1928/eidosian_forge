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
def test_output_stream_file_path(tmpdir):
    data = b'some test data\n' * 10 + b'eof\n'
    file_path = tmpdir / 'output_stream'

    def check_data(file_path, data):
        with pa.output_stream(file_path) as stream:
            stream.write(data)
        with open(str(file_path), 'rb') as f:
            assert f.read() == data
    check_data(file_path, data)
    check_data(str(file_path), data)
    check_data(pathlib.Path(str(file_path)), data)