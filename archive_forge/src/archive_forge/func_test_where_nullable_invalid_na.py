from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_nullable_invalid_na(frame_or_series, any_numeric_ea_dtype):
    arr = pd.array([1, 2, 3], dtype=any_numeric_ea_dtype)
    obj = frame_or_series(arr)
    mask = np.array([True, True, False], ndmin=obj.ndim).T
    msg = "Invalid value '.*' for dtype (U?Int|Float)\\d{1,2}"
    for null in tm.NP_NAT_OBJECTS + [pd.NaT]:
        with pytest.raises(TypeError, match=msg):
            obj.where(mask, null)
        with pytest.raises(TypeError, match=msg):
            obj.mask(mask, null)