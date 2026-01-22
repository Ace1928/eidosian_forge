from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_at_setitem_mixed_index_assignment(self):
    ser = Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 1, 2])
    ser.at['a'] = 11
    assert ser.iat[0] == 11
    ser.at[1] = 22
    assert ser.iat[3] == 22