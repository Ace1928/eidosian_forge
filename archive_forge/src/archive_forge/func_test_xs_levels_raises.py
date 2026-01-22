import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_levels_raises(self, frame_or_series):
    obj = DataFrame({'A': [1, 2, 3]})
    if frame_or_series is Series:
        obj = obj['A']
    msg = 'Index must be a MultiIndex'
    with pytest.raises(TypeError, match=msg):
        obj.xs(0, level='as')