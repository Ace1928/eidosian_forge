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
@pytest.mark.pandas
def test_read_single_file_list(tempdir):
    data_path = str(tempdir / 'data.parquet')
    table = pa.table({'a': [1, 2, 3]})
    _write_table(table, data_path)
    result = pq.ParquetDataset([data_path]).read()
    assert result.equals(table)