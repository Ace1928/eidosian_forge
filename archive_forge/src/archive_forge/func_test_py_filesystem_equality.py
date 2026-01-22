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
def test_py_filesystem_equality():
    handler1 = DummyHandler(1)
    handler2 = DummyHandler(2)
    handler3 = DummyHandler(2)
    fs1 = PyFileSystem(handler1)
    fs2 = PyFileSystem(handler1)
    fs3 = PyFileSystem(handler2)
    fs4 = PyFileSystem(handler3)
    assert fs2 is not fs1
    assert fs3 is not fs2
    assert fs4 is not fs3
    assert fs2 == fs1
    assert fs3 != fs2
    assert fs4 == fs3
    assert fs1 != LocalFileSystem()
    assert fs1 != object()