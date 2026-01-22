from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_bool_data_preserve_dtype(self):
    ser = Series([True, False, True])
    result = ser._get_bool_data()
    tm.assert_series_equal(result, ser)