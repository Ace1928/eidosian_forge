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
def test_read_column_duplicated_in_file(tempdir):
    table = pa.table([[1, 2, 3], [4, 5, 6], [7, 8, 9]], names=['a', 'b', 'a'])
    path = str(tempdir / 'data.feather')
    write_feather(table, path, version=2)
    result = read_table(path)
    assert result.equals(table)
    result = read_table(path, columns=[0, 2])
    assert result.column_names == ['a', 'a']
    with pytest.raises(ValueError):
        read_table(path, columns=['a', 'b'])