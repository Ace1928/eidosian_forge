import re
import sys
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('keep, expected', [('first', Series([False, False, True, False, True])), ('last', Series([True, True, False, False, False])), (False, Series([True, True, True, False, True]))])
def test_duplicated_keep(keep, expected):
    df = DataFrame({'A': [0, 1, 1, 2, 0], 'B': ['a', 'b', 'b', 'c', 'a']})
    result = df.duplicated(keep=keep)
    tm.assert_series_equal(result, expected)