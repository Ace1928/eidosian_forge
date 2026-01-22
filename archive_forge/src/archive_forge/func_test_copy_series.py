import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_copy_series(self):
    expected = Series([1, 2, 3])
    result = SimpleSeriesSubClass(expected).copy()
    tm.assert_series_equal(result, expected)