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
def test_series_none_color(self):
    series = Series([1, 2, 3])
    ax = series.plot(color=None)
    expected = _unpack_cycler(mpl.pyplot.rcParams)[:1]
    _check_colors(ax.get_lines(), linecolors=expected)