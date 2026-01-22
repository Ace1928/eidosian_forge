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
def test_dataset_partitioning(tempdir):
    import pyarrow.dataset as ds
    root_path = tempdir / 'test_partitioning'
    (root_path / '2012' / '10' / '01').mkdir(parents=True)
    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(table, str(root_path / '2012' / '10' / '01' / 'data.parquet'))
    part = ds.partitioning(field_names=['year', 'month', 'day'])
    result = pq.read_table(str(root_path), partitioning=part)
    assert result.column_names == ['a', 'year', 'month', 'day']
    result = pq.ParquetDataset(str(root_path), partitioning=part).read()
    assert result.column_names == ['a', 'year', 'month', 'day']