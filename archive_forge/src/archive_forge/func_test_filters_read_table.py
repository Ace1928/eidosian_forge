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
@pytest.mark.parametrize('filters', ([('integers', '<', 3)], [[('integers', '<', 3)]], pc.field('integers') < 3, pc.field('nested', 'a') < 3, pc.field('nested', 'b').cast(pa.int64()) < 3))
@pytest.mark.parametrize('read_method', ('read_table', 'read_pandas'))
def test_filters_read_table(tempdir, filters, read_method):
    read = getattr(pq, read_method)
    fs = LocalFileSystem._get_instance()
    base_path = tempdir
    integer_keys = [0, 1, 2, 3, 4]
    partition_spec = [['integers', integer_keys]]
    N = len(integer_keys)
    df = pd.DataFrame({'index': np.arange(N), 'integers': np.array(integer_keys, dtype='i4'), 'nested': np.array([{'a': i, 'b': str(i)} for i in range(N)])})
    _generate_partition_directories(fs, base_path, partition_spec, df)
    kwargs = dict(filesystem=fs, filters=filters)
    table = read(base_path, **kwargs)
    assert table.num_rows == 3