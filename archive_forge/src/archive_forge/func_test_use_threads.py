import io
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
@pytest.mark.pandas
def test_use_threads(version):
    num_values = (10, 10)
    path = random_path()
    TEST_FILES.append(path)
    values = np.random.randint(0, 10, size=num_values)
    columns = ['col_' + str(i) for i in range(10)]
    table = pa.Table.from_arrays(values, columns)
    write_feather(table, path, version=version)
    result = read_feather(path)
    assert_frame_equal(table.to_pandas(), result)
    result = read_feather(path, use_threads=False)
    assert_frame_equal(table.to_pandas(), result)
    result = read_table(path, use_threads=False)
    assert result.equals(table)