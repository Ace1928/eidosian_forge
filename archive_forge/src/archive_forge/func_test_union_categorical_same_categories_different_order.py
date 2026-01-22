import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categorical_same_categories_different_order(self):
    a = Series(Categorical(['a', 'b', 'c'], categories=['a', 'b', 'c']))
    b = Series(Categorical(['a', 'b', 'c'], categories=['b', 'a', 'c']))
    result = pd.concat([a, b], ignore_index=True)
    expected = Series(Categorical(['a', 'b', 'c', 'a', 'b', 'c'], categories=['a', 'b', 'c']))
    tm.assert_series_equal(result, expected)