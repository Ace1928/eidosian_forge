from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('breaks', [date_range('2018-01-01', periods=5), timedelta_range('0 days', periods=5)])
def test_maybe_convert_i8_nat(self, breaks):
    index = IntervalIndex.from_breaks(breaks)
    to_convert = breaks._constructor([pd.NaT] * 3).as_unit('ns')
    expected = Index([np.nan] * 3, dtype=np.float64)
    result = index._maybe_convert_i8(to_convert)
    tm.assert_index_equal(result, expected)
    to_convert = to_convert.insert(0, breaks[0])
    expected = expected.insert(0, float(breaks[0]._value))
    result = index._maybe_convert_i8(to_convert)
    tm.assert_index_equal(result, expected)