import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('op, expected', [['sum', np.int64(3)], ['prod', np.int64(2)], ['min', np.int64(1)], ['max', np.int64(2)], ['mean', np.float64(1.5)], ['median', np.float64(1.5)], ['var', np.float64(0.5)], ['std', np.float64(0.5 ** 0.5)], ['skew', pd.NA], ['kurt', pd.NA], ['any', True], ['all', True]])
def test_series_reductions(op, expected):
    ser = Series([1, 2], dtype='Int64')
    result = getattr(ser, op)()
    tm.assert_equal(result, expected)