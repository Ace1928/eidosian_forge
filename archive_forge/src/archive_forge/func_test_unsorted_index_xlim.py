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
def test_unsorted_index_xlim(self):
    ser = Series([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0], index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0])
    _, ax = mpl.pyplot.subplots()
    ax = ser.plot(ax=ax)
    xmin, xmax = ax.get_xlim()
    lines = ax.get_lines()
    assert xmin <= np.nanmin(lines[0].get_data(orig=False)[0])
    assert xmax >= np.nanmax(lines[0].get_data(orig=False)[0])