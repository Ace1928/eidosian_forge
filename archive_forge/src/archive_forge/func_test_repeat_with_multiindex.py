import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_repeat_with_multiindex(self):
    m_idx = MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6), (7, 8)])
    data = ['a', 'b', 'c', 'd']
    m_df = Series(data, index=m_idx)
    assert m_df.repeat(3).shape == (3 * len(data),)