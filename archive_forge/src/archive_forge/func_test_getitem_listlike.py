import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
@pytest.mark.parametrize('idx_type', [list, iter, Index, set, lambda keys: dict(zip(keys, range(len(keys)))), lambda keys: dict(zip(keys, range(len(keys)))).keys()], ids=['list', 'iter', 'Index', 'set', 'dict', 'dict_keys'])
@pytest.mark.parametrize('levels', [1, 2])
def test_getitem_listlike(self, idx_type, levels, float_frame):
    if levels == 1:
        frame, missing = (float_frame, 'food')
    else:
        frame = DataFrame(np.random.default_rng(2).standard_normal((8, 3)), columns=Index([('foo', 'bar'), ('baz', 'qux'), ('peek', 'aboo')], name=('sth', 'sth2')))
        missing = ('good', 'food')
    keys = [frame.columns[1], frame.columns[0]]
    idx = idx_type(keys)
    idx_check = list(idx_type(keys))
    if isinstance(idx, (set, dict)):
        with pytest.raises(TypeError, match='as an indexer is not supported'):
            frame[idx]
        return
    else:
        result = frame[idx]
    expected = frame.loc[:, idx_check]
    expected.columns.names = frame.columns.names
    tm.assert_frame_equal(result, expected)
    idx = idx_type(keys + [missing])
    with pytest.raises(KeyError, match='not in index'):
        frame[idx]