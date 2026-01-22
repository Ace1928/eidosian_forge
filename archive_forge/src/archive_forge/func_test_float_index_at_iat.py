from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_float_index_at_iat(self):
    ser = Series([1, 2, 3], index=[0.1, 0.2, 0.3])
    for el, item in ser.items():
        assert ser.at[el] == item
    for i in range(len(ser)):
        assert ser.iat[i] == i + 1