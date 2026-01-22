import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interpolate_akima(self):
    pytest.importorskip('scipy')
    ser = Series([10, 11, 12, 13])
    expected = Series([11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0], index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]))
    new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(float)
    interp_s = ser.reindex(new_index).interpolate(method='akima')
    tm.assert_series_equal(interp_s.loc[1:3], expected)
    expected = Series([11.0, 1.0, 1.0, 1.0, 12.0, 1.0, 1.0, 1.0, 13.0], index=Index([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]))
    new_index = ser.index.union(Index([1.25, 1.5, 1.75, 2.25, 2.5, 2.75])).astype(float)
    interp_s = ser.reindex(new_index).interpolate(method='akima', der=1)
    tm.assert_series_equal(interp_s.loc[1:3], expected)