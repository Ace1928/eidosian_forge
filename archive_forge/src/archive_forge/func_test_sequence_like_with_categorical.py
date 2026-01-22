import datetime
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_sequence_like_with_categorical(self):
    df = DataFrame({'id': [1, 2, 3, 4, 5, 6], 'raw_grade': ['a', 'b', 'b', 'a', 'a', 'e']})
    df['grade'] = Categorical(df['raw_grade'])
    result = list(df.grade.values)
    expected = np.array(df.grade.values).tolist()
    tm.assert_almost_equal(result, expected)
    for t in df.itertuples(index=False):
        str(t)
    for row, s in df.iterrows():
        str(s)
    for c, col in df.items():
        str(col)