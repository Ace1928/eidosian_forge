import sys
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import PYPY
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_access_by_position(index_flat):
    index = index_flat
    if len(index) == 0:
        pytest.skip("Test doesn't make sense on empty data")
    series = Series(index)
    assert index[0] == series.iloc[0]
    assert index[5] == series.iloc[5]
    assert index[-1] == series.iloc[-1]
    size = len(index)
    assert index[-1] == index[size - 1]
    msg = f'index {size} is out of bounds for axis 0 with size {size}'
    if is_dtype_equal(index.dtype, 'string[pyarrow]') or is_dtype_equal(index.dtype, 'string[pyarrow_numpy]'):
        msg = 'index out of bounds'
    with pytest.raises(IndexError, match=msg):
        index[size]
    msg = 'single positional indexer is out-of-bounds'
    with pytest.raises(IndexError, match=msg):
        series.iloc[size]