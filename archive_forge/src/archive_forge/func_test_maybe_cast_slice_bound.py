import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('make_range', [date_range, period_range])
def test_maybe_cast_slice_bound(self, make_range, frame_or_series):
    idx = make_range(start='2013/10/01', freq='D', periods=10)
    obj = DataFrame({'units': [100 + i for i in range(10)]}, index=idx)
    obj = tm.get_obj(obj, frame_or_series)
    msg = f'cannot do slice indexing on {type(idx).__name__} with these indexers \\[foo\\] of type str'
    with pytest.raises(TypeError, match=msg):
        idx._maybe_cast_slice_bound('foo', 'left')
    with pytest.raises(TypeError, match=msg):
        idx.get_slice_bound('foo', 'left')
    with pytest.raises(TypeError, match=msg):
        obj['2013/09/30':'foo']
    with pytest.raises(TypeError, match=msg):
        obj['foo':'2013/09/30']
    with pytest.raises(TypeError, match=msg):
        obj.loc['2013/09/30':'foo']
    with pytest.raises(TypeError, match=msg):
        obj.loc['foo':'2013/09/30']