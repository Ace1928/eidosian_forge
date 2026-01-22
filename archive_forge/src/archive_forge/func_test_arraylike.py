from datetime import time
import locale
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import Series
import pandas._testing as tm
from pandas.core.tools.times import to_time
def test_arraylike(self):
    arg = ['14:15', '20:20']
    expected_arr = [time(14, 15), time(20, 20)]
    assert to_time(arg) == expected_arr
    assert to_time(arg, format='%H:%M') == expected_arr
    assert to_time(arg, infer_time_format=True) == expected_arr
    assert to_time(arg, format='%I:%M%p', errors='coerce') == [None, None]
    msg = "errors='ignore' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = to_time(arg, format='%I:%M%p', errors='ignore')
    tm.assert_numpy_array_equal(res, np.array(arg, dtype=np.object_))
    msg = 'Cannot convert.+to a time with given format'
    with pytest.raises(ValueError, match=msg):
        to_time(arg, format='%I:%M%p', errors='raise')
    tm.assert_series_equal(to_time(Series(arg, name='test')), Series(expected_arr, name='test'))
    res = to_time(np.array(arg))
    assert isinstance(res, list)
    assert res == expected_arr