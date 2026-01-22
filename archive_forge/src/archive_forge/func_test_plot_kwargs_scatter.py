import numpy as np
import pytest
from pandas import (
from pandas.tests.plotting.common import (
def test_plot_kwargs_scatter(self):
    df = DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1, 2, 3, 2, 1], 'z': list('ababa')})
    res = df.groupby('z').plot.scatter(x='x', y='y')
    assert len(res['a'].collections) == 1