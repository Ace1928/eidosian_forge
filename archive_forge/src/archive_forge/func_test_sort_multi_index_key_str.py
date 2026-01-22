import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_multi_index_key_str(self):
    df = DataFrame({'a': ['B', 'a', 'C'], 'b': [0, 1, 0], 'c': list('abc'), 'd': [0, 1, 2]}).set_index(list('abc'))
    result = df.sort_index(level='a', key=lambda x: x.str.lower())
    expected = DataFrame({'a': ['a', 'B', 'C'], 'b': [1, 0, 0], 'c': list('bac'), 'd': [1, 0, 2]}).set_index(list('abc'))
    tm.assert_frame_equal(result, expected)
    result = df.sort_index(level=list('abc'), key=lambda x: x.str.lower() if x.name in ['a', 'c'] else -x)
    expected = DataFrame({'a': ['a', 'B', 'C'], 'b': [1, 0, 0], 'c': list('bac'), 'd': [1, 0, 2]}).set_index(list('abc'))
    tm.assert_frame_equal(result, expected)