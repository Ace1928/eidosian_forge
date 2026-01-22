from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
def test_parquet_read_from_buffer(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(table, str(tempdir / 'data.parquet'))
    with open(str(tempdir / 'data.parquet'), 'rb') as f:
        result = pq.read_table(f)
    assert result.equals(table)
    with open(str(tempdir / 'data.parquet'), 'rb') as f:
        result = pq.read_table(pa.PythonFile(f))
    assert result.equals(table)