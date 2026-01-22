import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_truncate_nonsortedindex(self, frame_or_series):
    obj = DataFrame({'A': ['a', 'b', 'c', 'd', 'e']}, index=[5, 3, 2, 9, 0])
    obj = tm.get_obj(obj, frame_or_series)
    msg = 'truncate requires a sorted index'
    with pytest.raises(ValueError, match=msg):
        obj.truncate(before=3, after=9)