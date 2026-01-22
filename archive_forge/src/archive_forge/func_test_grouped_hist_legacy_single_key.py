import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_grouped_hist_legacy_single_key(self):
    rs = np.random.default_rng(2)
    df = DataFrame(rs.standard_normal((10, 1)), columns=['A'])
    df['B'] = to_datetime(rs.integers(812419200000000000, 819331200000000000, size=10, dtype=np.int64))
    df['C'] = rs.integers(0, 4, 10)
    df['D'] = ['X'] * 10
    axes = df.hist(by='D', rot=30)
    _check_axes_shape(axes, axes_num=1, layout=(1, 1))
    _check_ticks_props(axes, xrot=30)