import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
def test_qcut_include_lowest():
    values = np.arange(10)
    ii = qcut(values, 4)
    ex_levels = IntervalIndex([Interval(-0.001, 2.25), Interval(2.25, 4.5), Interval(4.5, 6.75), Interval(6.75, 9)])
    tm.assert_index_equal(ii.categories, ex_levels)