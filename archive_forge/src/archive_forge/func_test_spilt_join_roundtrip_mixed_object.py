from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_spilt_join_roundtrip_mixed_object():
    ser = Series(['a_b', np.nan, 'asdf_cas_asdf', True, datetime.today(), 'foo', None, 1, 2.0])
    result = ser.str.split('_').str.join('_')
    expected = Series(['a_b', np.nan, 'asdf_cas_asdf', np.nan, np.nan, 'foo', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)