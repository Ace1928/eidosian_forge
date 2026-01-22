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
def test_read_column_selection(version):
    df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=['a', 'b', 'c'])
    _check_pandas_roundtrip(df, columns=['a', 'c'], expected=df[['a', 'c']], version=version)
    _check_pandas_roundtrip(df, columns=[0, 2], expected=df[['a', 'c']], version=version)
    _check_pandas_roundtrip(df, columns=['b', 'a'], expected=df[['b', 'a']], version=version)
    _check_pandas_roundtrip(df, columns=[1, 0], expected=df[['b', 'a']], version=version)