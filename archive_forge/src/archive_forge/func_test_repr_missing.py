import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('constructor,expected', [(Series, '(0.0, 1.0]    a\nNaN           b\n(2.0, 3.0]    c\ndtype: object'), (DataFrame, '            0\n(0.0, 1.0]  a\nNaN         b\n(2.0, 3.0]  c')])
def test_repr_missing(self, constructor, expected, using_infer_string, request):
    if using_infer_string and constructor is Series:
        request.applymarker(pytest.mark.xfail(reason='repr different'))
    index = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
    obj = constructor(list('abc'), index=index)
    result = repr(obj)
    assert result == expected