import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_scatter_colors(self):
    df = DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3]})
    with pytest.raises(TypeError, match='Specify exactly one of `c` and `color`'):
        df.plot.scatter(x='a', y='b', c='c', color='green')