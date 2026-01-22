import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.parquet
def test_open_dataset_filesystem(tempdir):
    table, path = _create_single_file(tempdir)
    dataset1 = ds.dataset(str(path))
    assert dataset1.schema.equals(table.schema)
    dataset2 = ds.dataset(str(path), filesystem=fs.LocalFileSystem())
    assert dataset2.schema.equals(table.schema)
    with change_cwd(tempdir):
        dataset3 = ds.dataset('test.parquet', filesystem=fs.LocalFileSystem())
    assert dataset3.schema.equals(table.schema)
    with pytest.raises(FileNotFoundError):
        ds.dataset(str(path), filesystem=fs._MockFileSystem())