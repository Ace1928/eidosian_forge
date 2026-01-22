from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_with_intervals(self):
    idx = IntervalIndex.from_breaks(np.arange(11), name='x')
    original = DataFrame({'x': idx, 'y': np.arange(10)})[['x', 'y']]
    result = original.set_index('x')
    expected = DataFrame({'y': np.arange(10)}, index=idx)
    tm.assert_frame_equal(result, expected)
    result2 = result.reset_index()
    tm.assert_frame_equal(result2, original)