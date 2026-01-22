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
@pytest.mark.xfail(raises=(TypeError, AssertionError), reason='Loss of type information in creation of categoricals.')
@pytest.mark.pandas
def test_filters_cutoff_exclusive_datetime(tempdir):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir
    date_keys = [datetime.date(2018, 4, 9), datetime.date(2018, 4, 10), datetime.date(2018, 4, 11), datetime.date(2018, 4, 12), datetime.date(2018, 4, 13)]
    partition_spec = [['dates', date_keys]]
    N = 5
    df = pd.DataFrame({'index': np.arange(N), 'dates': np.array(date_keys, dtype='datetime64')}, columns=['index', 'dates'])
    _generate_partition_directories(fs, base_path, partition_spec, df)
    dataset = pq.ParquetDataset(base_path, filesystem=fs, filters=[('dates', '<', '2018-04-12'), ('dates', '>', '2018-04-10')])
    table = dataset.read()
    result_df = table.to_pandas().sort_values(by='index').reset_index(drop=True)
    expected = pd.Categorical(np.array([datetime.date(2018, 4, 11)], dtype='datetime64'), categories=np.array(date_keys, dtype='datetime64'))
    assert result_df['dates'].values == expected