from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_category_numeric():
    a = Series(['a', 'b', 'c', 'd'])
    b = Series([1, 2, 3, 4], index=pd.CategoricalIndex(['b', 'c', 'd', 'e']))
    c = Series([1, 2, 3, 4], index=Index(['b', 'c', 'd', 'e']))
    exp = Series([np.nan, 1, 2, 3])
    tm.assert_series_equal(a.map(b), exp)
    exp = Series([np.nan, 1, 2, 3])
    tm.assert_series_equal(a.map(c), exp)