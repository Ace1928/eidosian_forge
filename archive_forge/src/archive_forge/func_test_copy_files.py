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
@pytest.mark.s3
def test_copy_files(s3_connection, s3fs, tempdir):
    fs = s3fs['fs']
    pathfn = s3fs['pathfn']
    path = pathfn('c.txt')
    with fs.open_output_stream(path) as f:
        f.write(b'test')
    host, port, access_key, secret_key = s3_connection
    source_uri = f's3://{access_key}:{secret_key}@{path}?scheme=http&endpoint_override={host}:{port}'
    local_path1 = str(tempdir / 'c_copied1.txt')
    copy_files(source_uri, local_path1)
    localfs = LocalFileSystem()
    with localfs.open_input_stream(local_path1) as f:
        assert f.read() == b'test'
    local_path2 = str(tempdir / 'c_copied2.txt')
    copy_files(path, local_path2, source_filesystem=fs)
    with localfs.open_input_stream(local_path2) as f:
        assert f.read() == b'test'
    local_path3 = str(tempdir / 'c_copied3.txt')
    destination_uri = _filesystem_uri(local_path3)
    copy_files(source_uri, destination_uri)
    with localfs.open_input_stream(local_path3) as f:
        assert f.read() == b'test'
    local_path4 = str(tempdir / 'c_copied4.txt')
    copy_files(source_uri, local_path4, destination_filesystem=localfs)
    with localfs.open_input_stream(local_path4) as f:
        assert f.read() == b'test'
    local_path5 = str(tempdir / 'c_copied5.txt')
    copy_files(source_uri, local_path5, chunk_size=1, use_threads=False)
    with localfs.open_input_stream(local_path5) as f:
        assert f.read() == b'test'