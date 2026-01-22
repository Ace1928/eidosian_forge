from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_cumcount_dupe_index(self):
    df = DataFrame([['a'], ['a'], ['a'], ['b'], ['a']], columns=['A'], index=[0] * 5)
    g = df.groupby('A')
    sg = g.A
    expected = Series([0, 1, 2, 0, 3], index=[0] * 5)
    tm.assert_series_equal(expected, g.cumcount())
    tm.assert_series_equal(expected, sg.cumcount())