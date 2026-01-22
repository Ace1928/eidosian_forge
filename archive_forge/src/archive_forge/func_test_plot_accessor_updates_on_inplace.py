from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.xfail(reason='GH#24426, see also github.com/pandas-dev/pandas/commit/ef1bd69fa42bbed5d09dd17f08c44fc8bfc2b685#r61470674')
def test_plot_accessor_updates_on_inplace(self):
    ser = Series([1, 2, 3, 4])
    _, ax = mpl.pyplot.subplots()
    ax = ser.plot(ax=ax)
    before = ax.xaxis.get_ticklocs()
    ser.drop([0, 1], inplace=True)
    _, ax = mpl.pyplot.subplots()
    after = ax.xaxis.get_ticklocs()
    tm.assert_numpy_array_equal(before, after)