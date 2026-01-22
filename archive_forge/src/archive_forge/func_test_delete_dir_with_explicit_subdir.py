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
def test_delete_dir_with_explicit_subdir(fs, pathfn):
    skip_fsspec_s3fs(fs)
    d = pathfn('directory/')
    nd = pathfn('directory/nested/')
    fs.create_dir(d)
    fs.create_dir(nd)
    fs.delete_dir(d)
    dir_info = fs.get_file_info(d)
    assert dir_info.type == FileType.NotFound
    d = pathfn('directory2')
    nd = pathfn('directory2/nested')
    f = pathfn('directory2/nested/target-file')
    fs.create_dir(d)
    fs.create_dir(nd)
    with fs.open_output_stream(f) as s:
        s.write(b'data')
    fs.delete_dir(d)
    dir_info = fs.get_file_info(d)
    assert dir_info.type == FileType.NotFound