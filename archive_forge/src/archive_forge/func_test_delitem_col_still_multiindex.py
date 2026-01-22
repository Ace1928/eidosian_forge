import re
import numpy as np
import pytest
from pandas import (
def test_delitem_col_still_multiindex(self):
    arrays = [['a', 'b', 'c', 'top'], ['', '', '', 'OD'], ['', '', '', 'wx']]
    tuples = sorted(zip(*arrays))
    index = MultiIndex.from_tuples(tuples)
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 4)), columns=index)
    del df['a', '', '']
    assert isinstance(df.columns, MultiIndex)