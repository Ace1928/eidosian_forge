from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import subprocess
import sys
import pytest
import weakref
import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
@pytest.mark.gzip
@pytest.mark.parametrize(('compression', 'buffer_size', 'compressor'), [(None, None, identity), (None, 64, identity), ('gzip', None, gzip.compress), ('gzip', 256, gzip.compress)])
def test_open_input_stream(fs, pathfn, compression, buffer_size, compressor):
    p = pathfn('open-input-stream')
    data = b'some data for reading\n' * 512
    with fs.open_output_stream(p) as s:
        s.write(compressor(data))
    with fs.open_input_stream(p, compression, buffer_size) as s:
        result = s.read()
    assert result == data