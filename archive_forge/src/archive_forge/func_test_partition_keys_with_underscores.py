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
def test_partition_keys_with_underscores(tempdir):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir
    string_keys = ['2019_2', '2019_3']
    partition_spec = [['year_week', string_keys]]
    N = 2
    df = pd.DataFrame({'index': np.arange(N), 'year_week': np.array(string_keys, dtype='object')}, columns=['index', 'year_week'])
    _generate_partition_directories(fs, base_path, partition_spec, df)
    dataset = pq.ParquetDataset(base_path)
    result = dataset.read()
    assert result.column('year_week').to_pylist() == string_keys