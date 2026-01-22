import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_times_array(self, times_frame):
    halflife = '23 days'
    times = times_frame.pop('C')
    gb = times_frame.groupby('A')
    result = gb.ewm(halflife=halflife, times=times).mean()
    expected = gb.ewm(halflife=halflife, times=times.values).mean()
    tm.assert_frame_equal(result, expected)