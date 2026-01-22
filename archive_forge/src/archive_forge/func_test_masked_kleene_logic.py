import builtins
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('data', [[False, False, False], [True, True, True], [pd.NA, pd.NA, pd.NA], [False, pd.NA, False], [True, pd.NA, True], [True, pd.NA, False]])
def test_masked_kleene_logic(bool_agg_func, skipna, data):
    ser = Series(data, dtype='boolean')
    expected_data = getattr(ser, bool_agg_func)(skipna=skipna)
    expected = Series(expected_data, index=np.array([0]), dtype='boolean')
    result = ser.groupby([0, 0, 0]).agg(bool_agg_func, skipna=skipna)
    tm.assert_series_equal(result, expected)