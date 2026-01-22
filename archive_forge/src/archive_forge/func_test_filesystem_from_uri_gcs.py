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
@pytest.mark.gcs
def test_filesystem_from_uri_gcs(gcs_server):
    from pyarrow.fs import GcsFileSystem
    host, port = gcs_server['connection']
    uri = 'gs://anonymous@' + f'mybucket/foo/bar?scheme=http&endpoint_override={host}:{port}&' + 'retry_limit_seconds=5&project_id=test-project-id'
    fs, path = FileSystem.from_uri(uri)
    assert isinstance(fs, GcsFileSystem)
    assert path == 'mybucket/foo/bar'
    fs.create_dir(path)
    [info] = fs.get_file_info([path])
    assert info.path == path
    assert info.type == FileType.Directory