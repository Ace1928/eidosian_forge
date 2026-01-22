from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_defaultdict_ignore_na():
    mapping = defaultdict(int, {1: 10, np.nan: 42})
    ser = Series([1, np.nan, 2])
    result = ser.map(mapping)
    expected = Series([10, 42, 0])
    tm.assert_series_equal(result, expected)