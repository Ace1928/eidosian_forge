from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_range_index_result(self):
    df1 = DataFrame({'a': [1, 2]})
    df2 = DataFrame({'b': [1, 2]})
    result = concat([df1, df2], sort=True, axis=1)
    expected = DataFrame({'a': [1, 2], 'b': [1, 2]})
    tm.assert_frame_equal(result, expected)
    expected_index = pd.RangeIndex(0, 2)
    tm.assert_index_equal(result.index, expected_index, exact=True)