import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('ignore_index', [True, False])
def test_explode_pyarrow_list_type(ignore_index):
    pa = pytest.importorskip('pyarrow')
    data = [[None, None], [1], [], [2, 3], None]
    ser = pd.Series(data, dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    result = ser.explode(ignore_index=ignore_index)
    expected = pd.Series(data=[None, None, 1, None, 2, 3, None], index=None if ignore_index else [0, 0, 1, 2, 3, 3, 4], dtype=pd.ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)