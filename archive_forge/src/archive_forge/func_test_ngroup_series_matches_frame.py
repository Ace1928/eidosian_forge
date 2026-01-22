from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ngroup_series_matches_frame(self):
    df = DataFrame({'A': list('aaaba')})
    s = Series(list('aaaba'))
    tm.assert_series_equal(df.groupby(s).ngroup(), s.groupby(s).ngroup())