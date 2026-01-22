import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('values', [DataFrame({'a': [1, 2, 3]}, dtype='category'), Series([1, 2, 3], dtype='category')])
def test_isin_category_frame(self, values):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    expected = DataFrame({'a': [True, True, True], 'b': [False, False, False]})
    result = df.isin(values)
    tm.assert_frame_equal(result, expected)