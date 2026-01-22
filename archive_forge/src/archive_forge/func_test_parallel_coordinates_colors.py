import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
@pytest.mark.parametrize('color', [('#556270', '#4ECDC4', '#C7F464'), ['dodgerblue', 'aquamarine', 'seagreen']])
def test_parallel_coordinates_colors(self, iris, color):
    from pandas.plotting import parallel_coordinates
    df = iris
    ax = _check_plot_works(parallel_coordinates, frame=df, class_column='Name', color=color)
    _check_colors(ax.get_lines()[:10], linecolors=color, mapping=df['Name'][:10])