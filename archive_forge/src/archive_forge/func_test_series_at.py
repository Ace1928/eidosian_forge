import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_series_at():
    arr = Categorical(['a', 'b', 'c'])
    ser = Series(arr)
    result = ser.at[0]
    assert result == 'a'