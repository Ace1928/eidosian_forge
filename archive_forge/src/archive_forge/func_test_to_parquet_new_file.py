import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_to_parquet_new_file(cleared_fs, df1):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip('fastparquet')
    df1.to_parquet('memory://test/test.csv', index=True, engine='fastparquet', compression=None)