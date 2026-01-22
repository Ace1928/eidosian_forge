import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('index', [Index([str(i) for i in range(20)]), date_range('2020-01-01', periods=20)])
def test_first_last_valid(self, index):
    mat = np.random.default_rng(2).standard_normal(len(index))
    mat[:5] = np.nan
    mat[-5:] = np.nan
    frame = DataFrame({'foo': mat}, index=index)
    assert frame.first_valid_index() == frame.index[5]
    assert frame.last_valid_index() == frame.index[-6]
    ser = frame['foo']
    assert ser.first_valid_index() == frame.index[5]
    assert ser.last_valid_index() == frame.index[-6]