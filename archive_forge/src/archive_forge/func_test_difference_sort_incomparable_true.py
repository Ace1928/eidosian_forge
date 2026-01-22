import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_difference_sort_incomparable_true():
    idx = MultiIndex.from_product([[1, pd.Timestamp('2000'), 2], ['a', 'b']])
    other = MultiIndex.from_product([[3, pd.Timestamp('2000'), 4], ['c', 'd']])
    msg = "'values' is not ordered, please explicitly specify the categories order "
    with pytest.raises(TypeError, match=msg):
        idx.difference(other, sort=True)