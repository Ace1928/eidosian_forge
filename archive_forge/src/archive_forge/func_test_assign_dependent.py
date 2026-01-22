import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_assign_dependent(self):
    df = DataFrame({'A': [1, 2], 'B': [3, 4]})
    result = df.assign(C=df.A, D=lambda x: x['A'] + x['C'])
    expected = DataFrame([[1, 3, 1, 2], [2, 4, 2, 4]], columns=list('ABCD'))
    tm.assert_frame_equal(result, expected)
    result = df.assign(C=lambda df: df.A, D=lambda df: df['A'] + df['C'])
    expected = DataFrame([[1, 3, 1, 2], [2, 4, 2, 4]], columns=list('ABCD'))
    tm.assert_frame_equal(result, expected)