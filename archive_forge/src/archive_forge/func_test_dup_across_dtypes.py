import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dup_across_dtypes(self):
    df = DataFrame([[1, 1, 1.0, 5], [1, 1, 2.0, 5], [2, 1, 3.0, 5]], columns=['foo', 'bar', 'foo', 'hello'])
    df['foo2'] = 7.0
    expected = DataFrame([[1, 1, 1.0, 5, 7.0], [1, 1, 2.0, 5, 7.0], [2, 1, 3.0, 5, 7.0]], columns=['foo', 'bar', 'foo', 'hello', 'foo2'])
    tm.assert_frame_equal(df, expected)
    result = df['foo']
    expected = DataFrame([[1, 1.0], [1, 2.0], [2, 3.0]], columns=['foo', 'foo'])
    tm.assert_frame_equal(result, expected)
    df['foo'] = 'string'
    expected = DataFrame([['string', 1, 'string', 5, 7.0], ['string', 1, 'string', 5, 7.0], ['string', 1, 'string', 5, 7.0]], columns=['foo', 'bar', 'foo', 'hello', 'foo2'])
    tm.assert_frame_equal(df, expected)
    del df['foo']
    expected = DataFrame([[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=['bar', 'hello', 'foo2'])
    tm.assert_frame_equal(df, expected)