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
def test_py_filesystem_ops():
    handler = DummyHandler()
    fs = PyFileSystem(handler)
    fs.create_dir('recursive', recursive=True)
    fs.create_dir('non-recursive', recursive=False)
    with pytest.raises(IOError):
        fs.create_dir('foobar')
    fs.delete_dir('delete_dir')
    fs.delete_dir_contents('delete_dir_contents')
    for path in ('', '/', '//'):
        with pytest.raises(ValueError):
            fs.delete_dir_contents(path)
        fs.delete_dir_contents(path, accept_root_dir=True)
    fs.delete_file('delete_file')
    fs.move('move_from', 'move_to')
    fs.copy_file('copy_file_from', 'copy_file_to')