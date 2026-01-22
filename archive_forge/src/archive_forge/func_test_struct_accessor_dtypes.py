import re
import pytest
from pandas.compat.pyarrow import (
from pandas import (
import pandas._testing as tm
def test_struct_accessor_dtypes():
    ser = Series([], dtype=ArrowDtype(pa.struct([('int_col', pa.int64()), ('string_col', pa.string()), ('struct_col', pa.struct([('int_col', pa.int64()), ('float_col', pa.float64())]))])))
    actual = ser.struct.dtypes
    expected = Series([ArrowDtype(pa.int64()), ArrowDtype(pa.string()), ArrowDtype(pa.struct([('int_col', pa.int64()), ('float_col', pa.float64())]))], index=Index(['int_col', 'string_col', 'struct_col']))
    tm.assert_series_equal(actual, expected)