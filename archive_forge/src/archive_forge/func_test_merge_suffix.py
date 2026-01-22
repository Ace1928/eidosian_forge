from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('col1, col2, kwargs, expected_cols', [(0, 0, {'suffixes': ('', '_dup')}, ['0', '0_dup']), (0, 0, {'suffixes': (None, '_dup')}, [0, '0_dup']), (0, 0, {'suffixes': ('_x', '_y')}, ['0_x', '0_y']), (0, 0, {'suffixes': ['_x', '_y']}, ['0_x', '0_y']), ('a', 0, {'suffixes': (None, '_y')}, ['a', 0]), (0.0, 0.0, {'suffixes': ('_x', None)}, ['0.0_x', 0.0]), ('b', 'b', {'suffixes': (None, '_y')}, ['b', 'b_y']), ('a', 'a', {'suffixes': ('_x', None)}, ['a_x', 'a']), ('a', 'b', {'suffixes': ('_x', None)}, ['a', 'b']), ('a', 'a', {'suffixes': (None, '_x')}, ['a', 'a_x']), (0, 0, {'suffixes': ('_a', None)}, ['0_a', 0]), ('a', 'a', {}, ['a_x', 'a_y']), (0, 0, {}, ['0_x', '0_y'])])
def test_merge_suffix(col1, col2, kwargs, expected_cols):
    a = DataFrame({col1: [1, 2, 3]})
    b = DataFrame({col2: [4, 5, 6]})
    expected = DataFrame([[1, 4], [2, 5], [3, 6]], columns=expected_cols)
    result = a.merge(b, left_index=True, right_index=True, **kwargs)
    tm.assert_frame_equal(result, expected)
    result = merge(a, b, left_index=True, right_index=True, **kwargs)
    tm.assert_frame_equal(result, expected)