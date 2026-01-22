import numpy as np
from pandas import (
import pandas._testing as tm
def test_infer_objects_interval(self, index_or_series):
    ii = interval_range(1, 10)
    obj = index_or_series(ii)
    result = obj.astype(object).infer_objects()
    tm.assert_equal(result, obj)