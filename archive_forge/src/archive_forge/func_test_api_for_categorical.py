import numpy as np
import pytest
from pandas import (
from pandas.core.strings.accessor import StringMethods
def test_api_for_categorical(any_string_method, any_string_dtype):
    s = Series(list('aabb'), dtype=any_string_dtype)
    s = s + ' ' + s
    c = s.astype('category')
    c = c.astype(CategoricalDtype(c.dtype.categories.astype('object')))
    assert isinstance(c.str, StringMethods)
    method_name, args, kwargs = any_string_method
    result = getattr(c.str, method_name)(*args, **kwargs)
    expected = getattr(s.astype('object').str, method_name)(*args, **kwargs)
    if isinstance(result, DataFrame):
        tm.assert_frame_equal(result, expected)
    elif isinstance(result, Series):
        tm.assert_series_equal(result, expected)
    else:
        assert result == expected