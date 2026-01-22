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
@pytest.fixture
def py_fsspec_s3fs(request, s3_server):
    s3fs = pytest.importorskip('s3fs')
    host, port, access_key, secret_key = s3_server['connection']
    bucket = 'pyarrow-filesystem/'
    fs = s3fs.S3FileSystem(key=access_key, secret=secret_key, client_kwargs=dict(endpoint_url='http://{}:{}'.format(host, port)))
    fs = PyFileSystem(FSSpecHandler(fs))
    fs.create_dir(bucket)
    yield dict(fs=fs, pathfn=bucket.__add__, allow_move_dir=False, allow_append_to_file=True)
    fs.delete_dir(bucket)