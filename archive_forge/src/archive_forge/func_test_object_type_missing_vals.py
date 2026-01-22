import builtins
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bool_agg_func,data,expected_res', [('any', [pd.NA, np.nan], False), ('any', [pd.NA, 1, np.nan], True), ('all', [pd.NA, pd.NaT], True), ('all', [pd.NA, False, pd.NaT], False)])
def test_object_type_missing_vals(bool_agg_func, data, expected_res, frame_or_series):
    obj = frame_or_series(data, dtype=object)
    result = obj.groupby([1] * len(data)).agg(bool_agg_func)
    expected = frame_or_series([expected_res], index=np.array([1]), dtype='bool')
    tm.assert_equal(result, expected)