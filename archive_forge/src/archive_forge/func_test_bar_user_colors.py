import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_bar_user_colors(self):
    df = DataFrame({'A': range(4), 'B': range(1, 5), 'color': ['red', 'blue', 'blue', 'red']})
    ax = df.plot.bar(y='A', color=df['color'])
    result = [p.get_facecolor() for p in ax.patches]
    expected = [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0)]
    assert result == expected