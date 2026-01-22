import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.s3
def test_read_directory_s3fs(s3_example_s3fs):
    fs, directory = s3_example_s3fs
    path = directory + '/test.parquet'
    table = pa.table({'a': [1, 2, 3]})
    _write_table(table, path, filesystem=fs)
    result = _read_table(directory, filesystem=fs)
    assert result.equals(table)