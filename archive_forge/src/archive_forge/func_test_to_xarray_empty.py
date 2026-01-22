import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_xarray_empty(self):
    from xarray import DataArray
    ser = Series([], dtype=object)
    ser.index.name = 'foo'
    result = ser.to_xarray()
    assert len(result) == 0
    assert len(result.coords) == 1
    tm.assert_almost_equal(list(result.coords.keys()), ['foo'])
    assert isinstance(result, DataArray)