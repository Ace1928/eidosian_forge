import pytest
from pandas import (
import pandas._testing as tm
def test_pipe_tuple_error(self, frame_or_series):
    obj = DataFrame({'A': [1, 2, 3]})
    obj = tm.get_obj(obj, frame_or_series)
    f = lambda x, y: y
    msg = 'y is both the pipe target and a keyword argument'
    with pytest.raises(ValueError, match=msg):
        obj.pipe((f, 'y'), x=1, y=0)