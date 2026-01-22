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
@pytest.mark.parametrize('data, index', [([1, 2, 3, 4], [3, 2, 1, 0]), ([10, 50, 20, 30], [1910, 1920, 1980, 1950])])
def test_plot_order(self, data, index):
    ser = Series(data=data, index=index)
    ax = ser.plot(kind='bar')
    expected = ser.tolist()
    result = [patch.get_bbox().ymax for patch in sorted(ax.patches, key=lambda patch: patch.get_bbox().xmax)]
    assert expected == result