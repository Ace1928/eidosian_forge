from itertools import product
import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('infer_objects, dtype', [(True, 'Int64'), (False, 'object')])
def test_convert_dtype_object_with_na(self, infer_objects, dtype):
    ser = pd.Series([1, pd.NA])
    result = ser.convert_dtypes(infer_objects=infer_objects)
    expected = pd.Series([1, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)