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
def test_output_stream_file_path_compressed(tmpdir):
    data = b'some test data\n' * 10 + b'eof\n'
    file_path = tmpdir / 'output_stream.gz'

    def check_data(file_path, data, **kwargs):
        with pa.output_stream(file_path, **kwargs) as stream:
            stream.write(data)
        with open(str(file_path), 'rb') as f:
            return f.read()
    assert gzip.decompress(check_data(file_path, data)) == data
    assert gzip.decompress(check_data(str(file_path), data)) == data
    assert gzip.decompress(check_data(pathlib.Path(str(file_path)), data)) == data
    assert gzip.decompress(check_data(file_path, data, compression='gzip')) == data
    assert check_data(file_path, data, compression=None) == data
    with pytest.raises(ValueError, match='Invalid value for compression'):
        assert check_data(file_path, data, compression='rabbit') == data