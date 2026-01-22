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
def test_partitioned_dataset(tempdir):
    path = tempdir / 'ARROW-3208'
    df = pd.DataFrame({'one': [-1, 10, 2.5, 100, 1000, 1, 29.2], 'two': [-1, 10, 2, 100, 1000, 1, 11], 'three': [0, 0, 0, 0, 0, 0, 0]})
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, root_path=str(path), partition_cols=['one', 'two'])
    table = pq.ParquetDataset(path).read()
    pq.write_table(table, path / 'output.parquet')