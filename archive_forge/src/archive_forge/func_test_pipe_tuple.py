import pytest
from pandas import (
import pandas._testing as tm
def test_pipe_tuple(self, frame_or_series):
    obj = DataFrame({'A': [1, 2, 3]})
    obj = tm.get_obj(obj, frame_or_series)
    f = lambda x, y: y
    result = obj.pipe((f, 'y'), 0)
    tm.assert_equal(result, obj)