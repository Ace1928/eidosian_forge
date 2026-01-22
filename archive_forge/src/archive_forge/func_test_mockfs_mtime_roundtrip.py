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
def test_mockfs_mtime_roundtrip(mockfs):
    dt = datetime.fromtimestamp(1568799826, timezone.utc)
    fs = _MockFileSystem(dt)
    with fs.open_output_stream('foo'):
        pass
    [info] = fs.get_file_info(['foo'])
    assert info.mtime == dt