from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_index_empty_column(self):
    df = DataFrame([{'a': 1, 'p': 0}, {'a': 2, 'm': 10}, {'a': 3, 'm': 11, 'p': 20}, {'a': 4, 'm': 12, 'p': 21}], columns=['a', 'm', 'p', 'x'])
    result = df.set_index(['a', 'x'])
    expected = df[['m', 'p']]
    expected.index = MultiIndex.from_arrays([df['a'], df['x']], names=['a', 'x'])
    tm.assert_frame_equal(result, expected)