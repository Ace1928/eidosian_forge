import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('time_dtype', ['m8[D]', 'M8[Y]'])
@pytest.mark.parametrize('str_dtype', ['U', 'S'])
def test_datetime_conversions_byteorders(self, str_dtype, time_dtype):
    times = np.array(['2017', 'NaT'], dtype=time_dtype)
    from_strings = np.array(['2017', 'NaT'], dtype=str_dtype)
    to_strings = times.astype(str_dtype)
    times_swapped = times.astype(times.dtype.newbyteorder())
    res = times_swapped.astype(str_dtype)
    assert_array_equal(res, to_strings)
    res = times_swapped.astype(to_strings.dtype.newbyteorder())
    assert_array_equal(res, to_strings)
    res = times.astype(to_strings.dtype.newbyteorder())
    assert_array_equal(res, to_strings)
    from_strings_swapped = from_strings.astype(from_strings.dtype.newbyteorder())
    res = from_strings_swapped.astype(time_dtype)
    assert_array_equal(res, times)
    res = from_strings_swapped.astype(times.dtype.newbyteorder())
    assert_array_equal(res, times)
    res = from_strings.astype(times.dtype.newbyteorder())
    assert_array_equal(res, times)