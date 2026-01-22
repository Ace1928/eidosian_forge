import numpy as np
from pandas import (
import pandas._testing as tm
def test_unique_categorical(self):
    cat = Categorical([])
    ser = Series(cat)
    result = ser.unique()
    tm.assert_categorical_equal(result, cat)
    cat = Categorical([np.nan])
    ser = Series(cat)
    result = ser.unique()
    tm.assert_categorical_equal(result, cat)