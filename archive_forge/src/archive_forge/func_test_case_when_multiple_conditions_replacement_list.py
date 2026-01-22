import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_case_when_multiple_conditions_replacement_list(df):
    """
    Test output when replacement is a list
    """
    result = Series([np.nan, np.nan, np.nan]).case_when([([True, False, False], 1), (df['a'].gt(1) & df['b'].eq(5), [1, 2, 3])])
    expected = Series([1, 2, np.nan])
    tm.assert_series_equal(result, expected)