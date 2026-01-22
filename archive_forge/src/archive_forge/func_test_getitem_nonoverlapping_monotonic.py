import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat import IS64
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('direction', ['increasing', 'decreasing'])
def test_getitem_nonoverlapping_monotonic(self, direction, closed, indexer_sl):
    tpls = [(0, 1), (2, 3), (4, 5)]
    if direction == 'decreasing':
        tpls = tpls[::-1]
    idx = IntervalIndex.from_tuples(tpls, closed=closed)
    ser = Series(list('abc'), idx)
    for key, expected in zip(idx.left, ser):
        if idx.closed_left:
            assert indexer_sl(ser)[key] == expected
        else:
            with pytest.raises(KeyError, match=str(key)):
                indexer_sl(ser)[key]
    for key, expected in zip(idx.right, ser):
        if idx.closed_right:
            assert indexer_sl(ser)[key] == expected
        else:
            with pytest.raises(KeyError, match=str(key)):
                indexer_sl(ser)[key]
    for key, expected in zip(idx.mid, ser):
        assert indexer_sl(ser)[key] == expected