import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_nat_na_position_default(self):
    expected = DataFrame({'A': [1, 2, 3, 4, 4], 'date': pd.DatetimeIndex(['2010-01-01 09:00:00', '2010-01-01 09:00:01', '2010-01-01 09:00:02', '2010-01-01 09:00:03', 'NaT'])})
    result = expected.sort_values(['A', 'date'])
    tm.assert_frame_equal(result, expected)