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
def test_output_stream_python_file(tmpdir):
    data = b'some test data\n' * 10 + b'eof\n'

    def check_data(data, **kwargs):
        fn = str(tmpdir / 'output_stream_file')
        with open(fn, 'wb') as f:
            with pa.output_stream(f, **kwargs) as stream:
                stream.write(data)
        with open(fn, 'rb') as f:
            return f.read()
    assert check_data(data) == data
    assert gzip.decompress(check_data(data, compression='gzip')) == data