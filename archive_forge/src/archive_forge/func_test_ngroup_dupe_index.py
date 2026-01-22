from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ngroup_dupe_index(self):
    df = DataFrame({'A': list('aaaba')}, index=[0] * 5)
    g = df.groupby('A')
    sg = g.A
    expected = Series([0, 0, 0, 1, 0], index=[0] * 5)
    tm.assert_series_equal(expected, g.ngroup())
    tm.assert_series_equal(expected, sg.ngroup())