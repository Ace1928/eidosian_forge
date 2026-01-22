import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
def test_slice_interval_step(self, series_with_interval_index):
    ser = series_with_interval_index.copy()
    msg = 'label-based slicing with step!=1 is not supported for IntervalIndex'
    with pytest.raises(ValueError, match=msg):
        ser[0:4:Interval(0, 1)]