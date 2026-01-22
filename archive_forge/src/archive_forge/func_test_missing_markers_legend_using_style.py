import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_missing_markers_legend_using_style(self):
    df = DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [2, 4, 1, 3, 2, 4], 'C': [3, 3, 2, 6, 4, 2], 'X': [1, 2, 3, 4, 5, 6]})
    _, ax = mpl.pyplot.subplots()
    for kind in 'ABC':
        df.plot('X', kind, label=kind, ax=ax, style='.')
    _check_legend_labels(ax, labels=['A', 'B', 'C'])
    _check_legend_marker(ax, expected_markers=['.', '.', '.'])