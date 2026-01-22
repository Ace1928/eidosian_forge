import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_duplicates_categorical_bool_na(self, nulls_fixture):
    ser = Series(Categorical([True, False, True, False, nulls_fixture], categories=[True, False], ordered=True))
    result = ser.drop_duplicates()
    expected = Series(Categorical([True, False, np.nan], categories=[True, False], ordered=True), index=[0, 1, 4])
    tm.assert_series_equal(result, expected)