import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_get_numeric_data_preserve_dtype(self):
    obj = DataFrame({'A': [1, '2', 3.0]}, columns=Index(['A'], dtype='object'))
    result = obj._get_numeric_data()
    expected = DataFrame(dtype=object, index=pd.RangeIndex(3), columns=[])
    tm.assert_frame_equal(result, expected)