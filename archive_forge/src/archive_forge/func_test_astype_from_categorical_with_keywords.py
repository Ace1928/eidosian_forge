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
def test_astype_from_categorical_with_keywords(self):
    lst = ['a', 'b', 'c', 'a']
    ser = Series(lst)
    exp = Series(Categorical(lst, ordered=True))
    res = ser.astype(CategoricalDtype(None, ordered=True))
    tm.assert_series_equal(res, exp)
    exp = Series(Categorical(lst, categories=list('abcdef'), ordered=True))
    res = ser.astype(CategoricalDtype(list('abcdef'), ordered=True))
    tm.assert_series_equal(res, exp)