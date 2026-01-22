from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_encode_errors_kwarg(any_string_dtype):
    ser = Series(['a', 'b', 'a\x9d'], dtype=any_string_dtype)
    msg = "'charmap' codec can't encode character '\\\\x9d' in position 1: character maps to <undefined>"
    with pytest.raises(UnicodeEncodeError, match=msg):
        ser.str.encode('cp1252')
    result = ser.str.encode('cp1252', 'ignore')
    expected = ser.map(lambda x: x.encode('cp1252', 'ignore'))
    tm.assert_series_equal(result, expected)