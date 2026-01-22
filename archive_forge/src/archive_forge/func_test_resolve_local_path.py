import pyarrow as pa
from pyarrow import filesystem
import os
import pytest
def test_resolve_local_path():
    for uri in ['/home/user/myfile.parquet', 'myfile.parquet', 'my # file ? parquet', 'C:/Windows/myfile.parquet', 'C:\\\\Windows\\\\myfile.parquet']:
        fs, path = filesystem.resolve_filesystem_and_path(uri)
        assert isinstance(fs, filesystem.LocalFileSystem)
        assert path == uri