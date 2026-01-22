import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_cut_out_of_range_more():
    name = 'x'
    ser = Series([0, -1, 0, 1, -3], name=name)
    ind = cut(ser, [0, 1], labels=False)
    exp = Series([np.nan, np.nan, np.nan, 0, np.nan], name=name)
    tm.assert_series_equal(ind, exp)