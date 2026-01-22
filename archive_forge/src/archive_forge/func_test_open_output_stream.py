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
@pytest.mark.parametrize(('compression', 'buffer_size', 'decompressor'), [(None, None, identity), (None, 64, identity), ('gzip', None, gzip.decompress), ('gzip', 256, gzip.decompress)])
def test_open_output_stream(fs, pathfn, compression, buffer_size, decompressor):
    p = pathfn('open-output-stream')
    data = b'some data for writing' * 1024
    with fs.open_output_stream(p, compression, buffer_size) as f:
        f.write(data)
    with fs.open_input_stream(p, compression, buffer_size) as f:
        assert f.read(len(data)) == data