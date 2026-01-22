from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_identical_different_object(self):
    df = DataFrame({'a': [1, 2]})
    ser = Series([3, 4])
    result, result2 = df.align(ser, axis=0)
    tm.assert_frame_equal(result, df)
    tm.assert_series_equal(result2, ser)
    assert df is not result
    assert ser is not result2