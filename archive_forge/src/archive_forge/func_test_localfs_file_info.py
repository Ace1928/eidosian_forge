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
def test_localfs_file_info(localfs):
    fs = localfs['fs']
    file_path = pathlib.Path(__file__)
    dir_path = file_path.parent
    [file_info, dir_info] = fs.get_file_info([file_path.as_posix(), dir_path.as_posix()])
    assert file_info.size == file_path.stat().st_size
    assert file_info.mtime_ns == file_path.stat().st_mtime_ns
    check_mtime(file_info)
    assert dir_info.mtime_ns == dir_path.stat().st_mtime_ns
    check_mtime(dir_info)