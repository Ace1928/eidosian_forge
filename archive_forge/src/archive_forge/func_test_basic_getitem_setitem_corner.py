from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_basic_getitem_setitem_corner(datetime_series):
    msg = 'key of type tuple not found and not a MultiIndex'
    with pytest.raises(KeyError, match=msg):
        datetime_series[:, 2]
    with pytest.raises(KeyError, match=msg):
        datetime_series[:, 2] = 2
    msg = 'Indexing with a single-item list'
    with pytest.raises(ValueError, match=msg):
        datetime_series[[slice(None, 5)]]
    result = datetime_series[slice(None, 5),]
    expected = datetime_series[:5]
    tm.assert_series_equal(result, expected)
    msg = "unhashable type(: 'slice')?"
    with pytest.raises(TypeError, match=msg):
        datetime_series[[5, [None, None]]]
    with pytest.raises(TypeError, match=msg):
        datetime_series[[5, [None, None]]] = 2