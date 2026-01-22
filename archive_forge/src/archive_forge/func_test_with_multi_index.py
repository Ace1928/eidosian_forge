import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_with_multi_index(self):
    index = date_range('1/1/2012', periods=4, freq='12h')
    index_as_arrays = [index.to_period(freq='D'), index.hour]
    s = Series([0, 1, 2, 3], index_as_arrays)
    assert isinstance(s.index.levels[0], PeriodIndex)
    assert isinstance(s.index.values[0][0], Period)