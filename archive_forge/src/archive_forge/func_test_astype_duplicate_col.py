import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_duplicate_col(self):
    a1 = Series([1, 2, 3, 4, 5], name='a')
    b = Series([0.1, 0.2, 0.4, 0.6, 0.8], name='b')
    a2 = Series([0, 1, 2, 3, 4], name='a')
    df = concat([a1, b, a2], axis=1)
    result = df.astype(str)
    a1_str = Series(['1', '2', '3', '4', '5'], dtype='str', name='a')
    b_str = Series(['0.1', '0.2', '0.4', '0.6', '0.8'], dtype=str, name='b')
    a2_str = Series(['0', '1', '2', '3', '4'], dtype='str', name='a')
    expected = concat([a1_str, b_str, a2_str], axis=1)
    tm.assert_frame_equal(result, expected)
    result = df.astype({'a': 'str'})
    expected = concat([a1_str, b, a2_str], axis=1)
    tm.assert_frame_equal(result, expected)