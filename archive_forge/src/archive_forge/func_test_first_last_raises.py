import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_first_last_raises(self, frame_or_series):
    obj = DataFrame([[1, 2, 3], [4, 5, 6]])
    obj = tm.get_obj(obj, frame_or_series)
    msg = "'first' only supports a DatetimeIndex index"
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg), pytest.raises(TypeError, match=msg):
        obj.first('1D')
    msg = "'last' only supports a DatetimeIndex index"
    with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg), pytest.raises(TypeError, match=msg):
        obj.last('1D')