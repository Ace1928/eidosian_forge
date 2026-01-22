import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_passed_bar_colors(self):
    color_tuples = [(0.9, 0, 0, 1), (0, 0.9, 0, 1), (0, 0, 0.9, 1)]
    colormap = mpl.colors.ListedColormap(color_tuples)
    barplot = DataFrame([[1, 2, 3]]).plot(kind='bar', cmap=colormap)
    assert color_tuples == [c.get_facecolor() for c in barplot.patches]