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
def test_parquet_dataset_new_filesystem(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(table, tempdir / 'data.parquet')
    filesystem = fs.SubTreeFileSystem(str(tempdir), fs.LocalFileSystem())
    dataset = pq.ParquetDataset('.', filesystem=filesystem)
    result = dataset.read()
    assert result.equals(table)