from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_tz(self, tz_aware_fixture):
    tz = tz_aware_fixture
    idx = date_range('1/1/2011', periods=5, freq='D', tz=tz, name='idx')
    df = DataFrame({'a': range(5), 'b': ['A', 'B', 'C', 'D', 'E']}, index=idx)
    expected = DataFrame({'idx': idx, 'a': range(5), 'b': ['A', 'B', 'C', 'D', 'E']}, columns=['idx', 'a', 'b'])
    result = df.reset_index()
    tm.assert_frame_equal(result, expected)