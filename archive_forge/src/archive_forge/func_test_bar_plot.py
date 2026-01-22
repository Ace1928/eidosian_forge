import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_bar_plot(self):
    from matplotlib.text import Text
    expected = [Text(0, 0, '0'), Text(1, 0, 'Total')]
    df = DataFrame({'a': [1, 2]}, index=Index([0, 'Total']))
    plot_bar = df.plot.bar()
    assert all((a.get_text() == b.get_text() for a, b in zip(plot_bar.get_xticklabels(), expected)))