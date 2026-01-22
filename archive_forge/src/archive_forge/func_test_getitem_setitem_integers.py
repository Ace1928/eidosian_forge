from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_getitem_setitem_integers():
    s = Series([1, 2, 3], ['a', 'b', 'c'])
    assert s.iloc[0] == s['a']
    s.iloc[0] = 5
    tm.assert_almost_equal(s['a'], 5)