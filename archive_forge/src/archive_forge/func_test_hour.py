import numpy as np
from pandas import (
import pandas._testing as tm
def test_hour(self):
    dr = date_range(start=Timestamp('2000-02-27'), periods=5, freq='h')
    r1 = Index([x.to_julian_date() for x in dr])
    r2 = dr.to_julian_date()
    assert isinstance(r2, Index) and r2.dtype == np.float64
    tm.assert_index_equal(r1, r2)