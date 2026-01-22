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
def test_pie_series_less_colors_than_elements(self):
    series = Series(np.random.default_rng(2).integers(1, 5), index=['a', 'b', 'c', 'd', 'e'], name='YLABEL')
    color_args = ['r', 'g', 'b']
    ax = _check_plot_works(series.plot.pie, colors=color_args)
    color_expected = ['r', 'g', 'b', 'r', 'g']
    _check_colors(ax.patches, facecolors=color_expected)