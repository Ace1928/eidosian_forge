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
def test_preserve_index_pandas(version):
    df = pd.DataFrame({'a': [1, 2, 3]}, index=['a', 'b', 'c'])
    if version == 1:
        expected = df.reset_index(drop=True).rename(columns=str)
    else:
        expected = df
    _check_pandas_roundtrip(df, expected, version=version)