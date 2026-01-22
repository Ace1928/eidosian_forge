import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_scatter_colors_white(self):
    df = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3]})
    ax = df.plot.scatter(x='a', y='b', color='white')
    tm.assert_numpy_array_equal(ax.collections[0].get_facecolor()[0], np.array([1, 1, 1, 1], dtype=np.float64))