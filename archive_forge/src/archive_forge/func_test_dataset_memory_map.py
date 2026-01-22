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
def test_dataset_memory_map(tempdir):
    dirpath = tempdir / guid()
    dirpath.mkdir()
    df = _test_dataframe(10, seed=0)
    path = dirpath / '{}.parquet'.format(0)
    table = pa.Table.from_pandas(df)
    _write_table(table, path, version='2.6')
    dataset = pq.ParquetDataset(dirpath, memory_map=True)
    assert dataset.read().equals(table)