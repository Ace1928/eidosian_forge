import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_bool_dtype
@pytest.mark.parametrize('data, index, drop_labels, axis, error_type, error_desc', [(range(3), list('abc'), 'bc', 0, KeyError, 'not found in axis'), (range(3), list('abc'), ('a',), 0, KeyError, 'not found in axis'), (range(3), list('abc'), 'one', 'columns', ValueError, 'No axis named columns')])
def test_drop_exception_raised(data, index, drop_labels, axis, error_type, error_desc):
    ser = Series(data, index=index)
    with pytest.raises(error_type, match=error_desc):
        ser.drop(drop_labels, axis=axis)