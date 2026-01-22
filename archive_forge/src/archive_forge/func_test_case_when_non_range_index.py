import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_case_when_non_range_index():
    """
    Test output if index is not RangeIndex
    """
    rng = np.random.default_rng(seed=123)
    dates = date_range('1/1/2000', periods=8)
    df = DataFrame(rng.standard_normal(size=(8, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
    result = Series(5, index=df.index, name='A').case_when([(df.A.gt(0), df.B)])
    expected = df.A.mask(df.A.gt(0), df.B).where(df.A.gt(0), 5)
    tm.assert_series_equal(result, expected)