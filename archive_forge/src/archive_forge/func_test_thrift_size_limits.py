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
def test_thrift_size_limits(tempdir):
    path = tempdir / 'largethrift.parquet'
    array = pa.array(list(range(10)))
    num_cols = 1000
    table = pa.table([array] * num_cols, names=[f'some_long_column_name_{i}' for i in range(num_cols)])
    pq.write_table(table, path)
    with pytest.raises(OSError, match="Couldn't deserialize thrift:.*Exceeded size limit"):
        pq.read_table(path, thrift_string_size_limit=50 * num_cols)
    with pytest.raises(OSError, match="Couldn't deserialize thrift:.*Exceeded size limit"):
        pq.read_table(path, thrift_container_size_limit=num_cols)
    got = pq.read_table(path, thrift_string_size_limit=100 * num_cols)
    assert got == table
    got = pq.read_table(path, thrift_container_size_limit=2 * num_cols)
    assert got == table
    got = pq.read_table(path)
    assert got == table