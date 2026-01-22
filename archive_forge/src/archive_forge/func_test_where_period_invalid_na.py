from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.mark.parametrize('as_cat', [True, False])
def test_where_period_invalid_na(frame_or_series, as_cat, request):
    idx = pd.period_range('2016-01-01', periods=3, freq='D')
    if as_cat:
        idx = idx.astype('category')
    obj = frame_or_series(idx)
    tdnat = pd.NaT.to_numpy('m8[ns]')
    mask = np.array([True, True, False], ndmin=obj.ndim).T
    if as_cat:
        msg = 'Cannot setitem on a Categorical with a new category \\(NaT\\), set the categories first'
    else:
        msg = "value should be a 'Period'"
    if as_cat:
        with pytest.raises(TypeError, match=msg):
            obj.where(mask, tdnat)
        with pytest.raises(TypeError, match=msg):
            obj.mask(mask, tdnat)
        with pytest.raises(TypeError, match=msg):
            obj.mask(mask, tdnat, inplace=True)
    else:
        expected = obj.astype(object).where(mask, tdnat)
        result = obj.where(mask, tdnat)
        tm.assert_equal(result, expected)
        expected = obj.astype(object).mask(mask, tdnat)
        result = obj.mask(mask, tdnat)
        tm.assert_equal(result, expected)
        with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
            obj.mask(mask, tdnat, inplace=True)
        tm.assert_equal(obj, expected)