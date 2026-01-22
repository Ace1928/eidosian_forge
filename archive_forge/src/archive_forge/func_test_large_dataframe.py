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
@pytest.mark.slow
@pytest.mark.pandas
def test_large_dataframe(version):
    df = pd.DataFrame({'A': np.arange(400000000)})
    _check_pandas_roundtrip(df, version=version)