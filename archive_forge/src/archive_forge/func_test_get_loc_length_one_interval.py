import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('other_closed', ['left', 'right', 'both', 'neither'])
@pytest.mark.parametrize('left, right', [(0, 5), (-1, 4), (-1, 6), (6, 7)])
def test_get_loc_length_one_interval(self, left, right, closed, other_closed):
    index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
    interval = Interval(left, right, closed=other_closed)
    if interval == index[0]:
        result = index.get_loc(interval)
        assert result == 0
    else:
        with pytest.raises(KeyError, match=re.escape(f"Interval({left}, {right}, closed='{other_closed}')")):
            index.get_loc(interval)