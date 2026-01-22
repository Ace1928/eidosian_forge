from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('items', [['a', 'b', 'c', 'a'], [1, 2, 3, 1]])
def test_astype_from_categorical(self, items):
    ser = Series(items)
    exp = Series(Categorical(items))
    res = ser.astype('category')
    tm.assert_series_equal(res, exp)