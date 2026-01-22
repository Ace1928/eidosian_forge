import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('first', ['m8[ns]', 'M8[ns]', 'M8[ns, US/Central]', 'period[D]'])
@pytest.mark.parametrize('second', ['m8[ns]', 'M8[ns]', 'M8[ns, US/Central]', 'period[D]'])
@pytest.mark.parametrize('box', [Series, Index, array])
def test_view_between_datetimelike(self, first, second, box):
    dti = date_range('2016-01-01', periods=3)
    orig = box(dti)
    obj = orig.view(first)
    assert obj.dtype == first
    tm.assert_numpy_array_equal(np.asarray(obj.view('i8')), dti.asi8)
    res = obj.view(second)
    assert res.dtype == second
    tm.assert_numpy_array_equal(np.asarray(obj.view('i8')), dti.asi8)