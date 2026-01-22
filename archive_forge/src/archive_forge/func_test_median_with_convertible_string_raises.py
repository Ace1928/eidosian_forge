import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_median_with_convertible_string_raises(using_array_manager):
    msg = "Cannot convert \\['1' '2' '3'\\] to numeric|does not support"
    ser = Series(['1', '2', '3'])
    with pytest.raises(TypeError, match=msg):
        ser.median()
    if not using_array_manager:
        msg = "Cannot convert \\[\\['1' '2' '3'\\]\\] to numeric|does not support"
    df = ser.to_frame()
    with pytest.raises(TypeError, match=msg):
        df.median()