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
@pytest.mark.parametrize('path', ['', '/', 'foo/bar', '/foo/bar', __file__])
def test_filesystem_from_path_object(path):
    p = pathlib.Path(path)
    fs, path = FileSystem.from_uri(p)
    assert isinstance(fs, LocalFileSystem)
    assert path == p.resolve().absolute().as_posix()