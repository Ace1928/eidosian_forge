import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multigroup(self, left, right):
    left = pd.concat([left, left], ignore_index=True)
    left['group'] = ['a'] * 3 + ['b'] * 3
    result = merge_ordered(left, right, on='key', left_by='group', fill_method='ffill')
    expected = DataFrame({'key': ['a', 'b', 'c', 'd', 'e', 'f'] * 2, 'lvalue': [1.0, 1, 2, 2, 3, 3.0] * 2, 'rvalue': [np.nan, 1, 2, 3, 3, 4] * 2})
    expected['group'] = ['a'] * 6 + ['b'] * 6
    tm.assert_frame_equal(result, expected.loc[:, result.columns])
    result2 = merge_ordered(right, left, on='key', right_by='group', fill_method='ffill')
    tm.assert_frame_equal(result, result2.loc[:, result.columns])
    result = merge_ordered(left, right, on='key', left_by='group')
    assert result['group'].notna().all()