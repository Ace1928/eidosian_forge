from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_putmask_keep_dtype(self, any_numeric_ea_dtype):
    midx = MultiIndex.from_arrays([pd.Series([1, 2, 3], dtype=any_numeric_ea_dtype), [10, 11, 12]])
    midx2 = MultiIndex.from_arrays([pd.Series([5, 6, 7], dtype=any_numeric_ea_dtype), [-1, -2, -3]])
    result = midx.putmask([True, False, False], midx2)
    expected = MultiIndex.from_arrays([pd.Series([5, 2, 3], dtype=any_numeric_ea_dtype), [-1, 11, 12]])
    tm.assert_index_equal(result, expected)